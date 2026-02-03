import json
from collections.abc import Generator
from typing import TYPE_CHECKING
import os
import gradio as gr
from datetime import datetime
from typing import Optional, Dict, List
import uuid
import asyncio
import time
from typing import Any
import logging
import shutil

from gradio.components import Component
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.agent.service import Agent
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from src.agent.deep_research.deep_research_agent import DeepResearchAgent

logger = logging.getLogger(__name__)

class WebuiManager:
    def __init__(self, settings_save_dir: str = "./tmp/webui_settings"):
        self.id_to_component: dict[str, Component] = {}
        self.component_to_id: dict[Component, str] = {}

        self.settings_save_dir = settings_save_dir
        os.makedirs(self.settings_save_dir, exist_ok=True)

    def init_browser_use_agent(self) -> None:
        """
        init browser use agent
        """
        self.bu_agent: Optional[Agent] = None
        self.bu_browser: Optional[CustomBrowser] = None
        self.bu_browser_context: Optional[CustomBrowserContext] = None
        self.bu_controller: Optional[CustomController] = None
        self.bu_chat_history: List[Dict[str, Optional[str]]] = []
        self.bu_response_event: Optional[asyncio.Event] = None
        self.bu_user_help_response: Optional[str] = None
        self.bu_current_task: Optional[asyncio.Task] = None
        self.bu_agent_task_id: Optional[str] = None
        self.bu_agent_status: str = "Ready"
        self.bu_latest_screenshot: Optional[str] = None
        self.bu_last_task_prompt: Optional[str] = None
        self.bu_max_steps: int = 100

    def init_deep_research_agent(self) -> None:
        """
        init deep research agent
        """
        self.dr_agent: Optional[DeepResearchAgent] = None
        self.dr_current_task = None
        self.dr_agent_task_id: Optional[str] = None
        self.dr_save_dir: Optional[str] = None

    def init_enhanced_agent(self) -> None:
        """
        init enhanced agent
        """
        self.enhanced_agent: Optional[Any] = None # Using Any to avoid circular import
        self.enhanced_agent_task: Optional[asyncio.Task] = None

    def add_components(self, tab_name: str, components_dict: dict[str, "Component"]) -> None:
        """
        Add tab components
        """
        for comp_name, component in components_dict.items():
            comp_id = f"{tab_name}.{comp_name}"
            self.id_to_component[comp_id] = component
            self.component_to_id[component] = comp_id

    def get_components(self) -> list["Component"]:
        """
        Get all components
        """
        return list(self.id_to_component.values())

    def get_component_by_id(self, comp_id: str) -> "Component":
        """
        Get component by id
        """
        return self.id_to_component[comp_id]

    def get_id_by_component(self, comp: "Component") -> str:
        """
        Get id by component
        """
        return self.component_to_id[comp]

    def save_config(self, components: Dict["Component", str]) -> None:
        """
        Save config
        """
        cur_settings = {}
        for comp in components:
            if not isinstance(comp, gr.Button) and not isinstance(comp, gr.File) and str(
                    getattr(comp, "interactive", True)).lower() != "false":
                comp_id = self.get_id_by_component(comp)
                cur_settings[comp_id] = components[comp]

        config_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_path = os.path.join(self.settings_save_dir, f"{config_name}.json")
        temp_path = f"{file_path}.tmp"

        try:
            with open(temp_path, "w") as fw:
                json.dump(cur_settings, fw, indent=4)
            
            # Retry logic for file operations on Windows
            for i in range(3):
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    os.rename(temp_path, file_path)
                    break
                except OSError:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to save config to {file_path}: {e}")
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

        return os.path.join(self.settings_save_dir, f"{config_name}.json")

    def load_config(self, config_path: str):
        """
        Load config
        """
        if not config_path:
            return

        try:
            with open(config_path, "r") as fr:
                ui_settings = json.load(fr)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return

        update_components = {}
        for comp_id, comp_val in ui_settings.items():
            if comp_id in self.id_to_component:
                comp = self.id_to_component[comp_id]

                # Skip buttons to prevent overwriting labels
                if isinstance(comp, gr.Button):
                    continue

                if comp_val is None and isinstance(comp, (gr.Slider, gr.Number)):
                    # Fallback to default value or 0 if None to prevent TypeError
                    if hasattr(comp, 'value') and comp.value is not None:
                        comp_val = comp.value
                    else:
                        comp_val = 0

                if comp.__class__.__name__ == "Chatbot":
                    update_components[comp] = gr.update(value=comp_val, type="messages")
                else:
                    update_components[comp] = gr.update(value=comp_val)
                    if comp_id in ["agent_settings.planner_llm_provider", "agent_settings.confirmer_llm_provider"]:
                        yield update_components  # yield provider, let callback run
                        time.sleep(0.1)  # wait for Gradio UI callback

        config_status = self.id_to_component["load_save_config.config_status"]
        update_components.update(
            {
                config_status: gr.update(value=f"Successfully loaded config: {config_path}")
            }
        )
        yield update_components

    def update_parameter(self, comp: Component, value: Any):
        """
        Update a single parameter in the config file
        """
        comp_id = self.get_id_by_component(comp)
        config_path = os.path.join(self.settings_save_dir, "last_config.json")
        temp_path = f"{config_path}.tmp"
        
        settings = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    settings = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read existing config to update parameter: {e}")
                # If the file is corrupted, we might start fresh or backup. 
                # For now, starting fresh settings dict is safer than crashing.
        
        settings[comp_id] = value
        
        try:
            with open(temp_path, "w") as f:
                json.dump(settings, f, indent=4)
            
            # Retry logic for file operations on Windows
            for i in range(3):
                try:
                    if os.path.exists(config_path):
                        os.remove(config_path)
                    os.rename(temp_path, config_path)
                    break
                except OSError:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to update parameter in config: {e}")

    def load_last_config(self):
        """Load the last saved config"""
        config_path = os.path.join(self.settings_save_dir, "last_config.json")
        if os.path.exists(config_path):
            yield from self.load_config(config_path)
