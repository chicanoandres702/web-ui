import logging
import json
import os
from typing import Any, Dict
from browser_use.browser.context import BrowserContext
from src.utils.utils import save_text_to_file as utils_save_text, sanitize_filename
from src.utils.memory_utils import reset_mem0, get_memory_manager
from src.utils.knowledge_base import search_kb_files, load_kb_content, list_kb_files
from src.controller.helpers import download_resource
from src.utils.browser_scripts import JS_GET_PAGE_METADATA, JS_EXTRACT_LINKS, JS_GET_MAIN_CONTENT

logger = logging.getLogger(__name__)

class SystemActionsMixin:
    """
    Mixin providing system-level actions. 
    Expected to be mixed into a class that has a 'registry' attribute (ActionRegistry).
    """
    # Type hints for attributes expected from the base class
    registry: Any
    webui_manager: Any
    kb_dir: str
    callbacks: Dict[str, Any]

    def _register_system_actions(self):
        @self.registry.action("Update the HUD (Heads Up Display) overlay on the page") # type: ignore
        async def update_hud(browser: BrowserContext, data_json: str):
            try:
                data = json.loads(data_json)
                return await self._update_hud_impl(browser, data)
            except Exception as e:
                return f"Error parsing HUD data: {e}"

        @self.registry.action("Add a new step to the plan. Optional: provide 'after_index' to insert after a specific step.") # type: ignore
        async def add_plan_step(browser: BrowserContext, step_description: str, after_index: int = None):
            if self.webui_manager and hasattr(self.webui_manager, "bu_plan"):
                actual_index = len(self.webui_manager.bu_plan)
                if after_index is not None:
                    idx = int(after_index)
                    if idx < 0: idx = 0
                    if idx > len(self.webui_manager.bu_plan): idx = len(self.webui_manager.bu_plan)
                    actual_index = idx + 1
                    self.webui_manager.add_plan_step(step_description, index=idx)
                else:
                    self.webui_manager.add_plan_step(step_description)

                
                msg = f"Added step {actual_index}: '{step_description}'."
                
                await self.refresh_hud(browser, last_action=msg)
                return msg
            return "Plan manager not available."

        @self.registry.action("Update the status of a plan step. Status options: 'pending', 'in_progress', 'completed', 'failed'") # type: ignore
        async def update_plan_step(browser: BrowserContext, step_index: int, status: str, result: str = None):
            if not (self.webui_manager and hasattr(self.webui_manager, "bu_plan")):
                return "Plan manager not available."

            try:
                step_index = int(step_index)
            except ValueError:
                return f"Invalid step_index: {step_index}. Must be an integer."

            plan = self.webui_manager.bu_plan
            idx = step_index - 1  # Convert 1-based to 0-based

            if not (0 <= idx < len(plan)):
                return f"Step index {step_index} out of bounds."

            # Get step description for feedback
            step_desc_current = plan[idx].get("step", "Unknown Step")

            # Update status
            self.webui_manager.update_plan_step_status(idx, status, result=result)

            # Trigger callback if registered
            if hasattr(self, "callbacks") and "update_plan" in self.callbacks:
                try:
                    await self.callbacks["update_plan"](step_index, status)
                except Exception as e:
                    logger.warning(f"Error in update_plan callback: {e}")

            # Construct helpful return message for the agent
            msg = f"Updated step {step_index} ('{step_desc_current}') status to '{status}'."
            if result:
                msg += f" Result: {result}"

            if status == "completed":
                # Find next pending step
                next_step_idx = -1
                for i in range(idx + 1, len(plan)):
                    if plan[i]["status"] == "pending":
                        next_step_idx = i
                        break

                if next_step_idx != -1:
                    next_step_num = next_step_idx + 1
                    next_step_desc = plan[next_step_idx]['step']
                    msg += f"\n👉 NEXT STEP ({next_step_num}): {next_step_desc}\nProceed to this step immediately."
                else:
                    msg += "\n✅ All steps completed. Verify goal and finish."
            elif status == "failed":
                msg += "\n❌ Step failed. Re-assess strategy or update plan."

            await self.refresh_hud(browser, last_action=msg)

            return msg

        @self.registry.action("Mark a specific step in the plan as completed") # type: ignore
        async def mark_step_complete(browser: BrowserContext, step_index: int):
            # Alias to update_plan_step for backward compatibility
            return await update_plan_step(browser, step_index, "completed")

        @self.registry.action("Mark the currently active step as completed and proceed to the next.") # type: ignore
        async def complete_current_step(browser: BrowserContext):
            """
            Automatically identifies the step marked 'in_progress', marks it as 'completed',
            and returns the instruction for the next step.
            """
            if not (self.webui_manager and hasattr(self.webui_manager, "bu_plan")):
                return "Plan manager not available."
            
            plan = self.webui_manager.bu_plan
            current_idx = -1
            
            for i, step in enumerate(plan):
                if step.get("status") == "in_progress":
                    current_idx = i
                    break
            
            if current_idx == -1:
                return "No step is currently 'in_progress'. Please use update_plan_step to start a step."

            return await update_plan_step(browser, current_idx + 1, "completed")

        @self.registry.action("Save text content to a file") # type: ignore
        async def save_text_to_file(filename: str, content: str, append: bool = False):
            safe_dir = os.path.abspath("./tmp/downloads")
            os.makedirs(safe_dir, exist_ok=True)
            if not os.path.exists(safe_dir):
                 logger.error(f"Could not create save directory at {safe_dir}")
            filepath = os.path.join(safe_dir, os.path.basename(filename))

            mode = "a" if append else "w"
            utils_save_text(filepath, content, mode)
            return f"Saved content to {filepath}"

        @self.registry.action("Clear browser cookies and local storage") # type: ignore
        async def clear_browser_data(browser: BrowserContext):
            page = await browser.get_current_page()
            try:
                context = page.context
                await context.clear_cookies()
                await page.evaluate("try { window.localStorage.clear(); window.sessionStorage.clear(); } catch (e) {}")
                return "Cleared cookies and local/session storage."
            except Exception as e:
                return f"Error clearing browser data: {e}"

        @self.registry.action("Clear agent memory (Mem0)") # type: ignore
        async def clear_agent_memory():
            memory_manager = get_memory_manager()
            if memory_manager and hasattr(memory_manager, 'memory_dir'):
                 logger.info(f"Clearing Agent Memory at {memory_manager.memory_dir}")
            reset_mem0()
            return "Agent memory (Mem0) cleared."

        @self.registry.action("Save specific navigation knowledge for a website") # type: ignore
        async def save_site_knowledge(browser: BrowserContext, domain: str, knowledge: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            # Use the shared memory manager for consistency
            manager = get_memory_manager(self.kb_dir)
            
            if manager.save_site_knowledge(domain, knowledge):
                 return f"Saved knowledge for {domain}."
            return f"Failed to save knowledge for {domain}."

        @self.registry.action("Automatically learn and save the page's topic and navigation structure") # type: ignore
        async def learn_page_topic_and_navigation(browser: BrowserContext):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            page = await browser.get_current_page()
            from urllib.parse import urlparse
            
            try:
                # 1. Get Metadata
                metadata = await page.evaluate(JS_GET_PAGE_METADATA)
                
                # 2. Get Main Content Summary
                content = await page.evaluate(JS_GET_MAIN_CONTENT)
                summary = content[:500].replace("\n", " ") + "..." if len(content) > 500 else content
                
                # 3. Get Navigation Links (Top 20 relevant)
                links = await page.evaluate(JS_EXTRACT_LINKS)
                # Filter for nav/header links
                nav_links = [l for l in links if l.get('context') in ['nav', 'header']]
                if not nav_links:
                    nav_links = links[:10] # Fallback
                else:
                    nav_links = nav_links[:20]
                
                nav_summary = ", ".join([f"[{l['text']}]({l['href']})" for l in nav_links])
                
                # 4. Construct Knowledge Entry
                domain = urlparse(page.url).netloc.replace("www.", "")
                knowledge_entry = (
                    f"**Topic**: {metadata.get('title', 'Unknown')}\n"
                    f"**Description**: {metadata.get('description', 'N/A')}\n"
                    f"**Summary**: {summary}\n"
                    f"**Navigation**: {nav_summary}"
                )
                
                # 5. Save
                manager = get_memory_manager(self.kb_dir)
                if manager.save_site_knowledge(domain, knowledge_entry):
                    return f"Learned and saved knowledge for {domain}:\n{knowledge_entry}"
                return f"Failed to save learned knowledge for {domain}."
                
            except Exception as e:
                return f"Error learning from page: {e}"

        @self.registry.action("Save general knowledge/study notes to the knowledge base") # type: ignore
        async def save_to_knowledge_base(filename: str, content: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            os.makedirs(self.kb_dir, exist_ok=True)
            
            if not (filename.endswith('.md') or filename.endswith('.txt')):
                filename += '.md'
            
            safe_filename = sanitize_filename(filename)
            filepath = os.path.join(self.kb_dir, safe_filename)
            if not os.path.exists(self.kb_dir):
                 logger.error(f"Knowledge base directory does not exist: {self.kb_dir}")
            utils_save_text(filepath, content, "w")
            return f"Saved knowledge to {filepath}"

        @self.registry.action("Retrieve stored knowledge for a website") # type: ignore
        async def get_site_knowledge(browser: BrowserContext, domain: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            manager = get_memory_manager(self.kb_dir)
            content = manager.get_site_knowledge(domain)
            
            if content:
                return f"Stored Knowledge for {domain}:\n{content}"
            return f"No specific knowledge found for {domain}."

        @self.registry.action("Download a file from a URL") # type: ignore
        async def download_file(browser: BrowserContext, url: str, filename: str):
            path = browser.config.save_downloads_path
            if not path:
                return "Error: No download path configured."
            
            os.makedirs(path, exist_ok=True)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')).strip()
            filepath = os.path.join(path, safe_filename)
            
            page = await browser.get_current_page()
            
            try:
                return await download_resource(page, url, filepath)
            except Exception as e:
                return f"Error downloading file: {e}"

        @self.registry.action("Search the knowledge base for information") # type: ignore
        async def search_knowledge_base(query: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            matches = search_kb_files(self.kb_dir, query)
            if not matches:
                return f"No matches found in knowledge base for query: '{query}'"
            
            # If few matches, return content of first few
            result = f"Found {len(matches)} matches:\n"
            for filename in matches[:3]: # Limit to top 3
                content = load_kb_content(self.kb_dir, filename)
                preview = content[:500] + "..." if len(content) > 500 else content
                result += f"\n--- File: {filename} ---\n{preview}\n"
            
            if len(matches) > 3:
                result += f"\n...and {len(matches) - 3} more files: {', '.join(matches[3:])}"
                
            return result

        @self.registry.action("Read the full content of a specific knowledge base file") # type: ignore
        async def read_knowledge_base_file(filename: str):
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            content = load_kb_content(self.kb_dir, filename)
            if not content:
                return f"File '{filename}' not found or empty."
            
            return f"--- File: {filename} ---\n{content}"

        @self.registry.action("List all available knowledge base files") # type: ignore
        async def list_knowledge_base_files():
            if not self.kb_dir:
                return "Knowledge base directory not configured."
            
            files = list_kb_files(self.kb_dir)
            if not files:
                return "No knowledge base files found."
            
            return f"Available Knowledge Base Files:\n" + "\n".join(files)
