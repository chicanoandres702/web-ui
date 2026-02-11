from __future__ import annotations
import asyncio
from asyncio.log import logger
import inspect
from queue import Empty
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, cast
import json
import json_repair
import logging
import os
import re
from pathlib import Path
import sys
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langchain_core.runnables import Runnable, RunnableLambda

from browser_use.agent.views import AgentOutput
from src.utils.utils import parse_json_safe, retry_async
from src.utils.prompts import CONFIRMER_PROMPT_FAST, CONFIRMER_PROMPT_STANDARD


class AgentLLMManager:
    """Manages LLM-related operations for the agent, including model switching and patching."""

    def __init__(self, agent: Any, model_priority_list: Optional[List[Dict]] = None):
        self.agent = agent
        self.model_priority_list = model_priority_list

    def set_llm(self, llm):
        """Sets the current LLM for the agent."""
        self.agent.llm = llm

    def get_current_llm(self):
        """Returns the current LLM."""
        return self.agent.llm

    def _patch_llm_with_qwen_fix(self, llm: Runnable) -> Runnable:
        """Wraps the LLM with a RunnableLambda that applies Qwen/Ollama JSON fix logic."""
        llm_with_aimessage_output = llm | RunnableLambda(
            self._convert_agent_output_to_aimessage
        )
        return llm_with_aimessage_output | RunnableLambda(self._apply_qwen_fix_logic)

    def _convert_agent_output_to_aimessage(
        self, agent_output: AgentOutput
    ) -> BaseMessage:
        tool_calls = []
        json_content = agent_output.model_dump_json()
        if agent_output.action:
            actions_list: List[Any] = (
                agent_output.action
                if isinstance(agent_output.action, list)
                else [agent_output.action]
            )
            for action_model in actions_list:
                if hasattr(action_model, "model_dump"):
                    action_args = action_model.model_dump(exclude_unset=True)
                    tool_name = action_args.pop("name", None)
                    if tool_name:
                        tool_calls.append({"name": tool_name, "args": action_args})
        return AIMessage(content=json_content, tool_calls=tool_calls)

    def _apply_qwen_fix_logic(self, response: BaseMessage) -> BaseMessage:
        """Applies the Qwen/Ollama JSON fix logic to an LLM response (BaseMessage)."""
        if not hasattr(response, "content") or not isinstance(response.content, str):
            return response

        return response

    def _set_tool_calling_method(self) -> str | None:  # type: ignore
        """Sets the tool calling method based on model and configuration."""
        settings = getattr(self.agent, "settings", None)
        if not settings:
            return None

        tool_calling_method = getattr(settings, "tool_calling_method", "auto")
        chat_model_library = getattr(self.agent, "chat_model_library", "")
        model_name = getattr(self.agent, "model_name", "").lower()

        if tool_calling_method == "auto":
            if "Ollama" in chat_model_library:
                return None
            elif (
                any(m in model_name for m in ["qwen", "deepseek", "gpt", "claude"])
                and "Ollama" not in chat_model_library
            ):
                return "function_calling"
            return None
        else:
            return tool_calling_method

    async def request_user_decision(
        self, intel: str, next_task: str
    ) -> Union[bool, str]:  # type: ignore
        """
        Requests confirmation/input from the user via WebSocket.
        """
        if (
            not self.agent.send_agent_message_callback
            or not self.agent.confirmation_response_queue
        ):
            return True

        await self.agent.send_agent_message_callback(
            {
                "type": "request_confirmation",
                "message": "Agent is pausing for your review.",
                "intel": intel,
                "next_task": next_task,
            }
        )

        while True:
            # Create tasks for both queues
            confirmation_task = asyncio.create_task(
                self.agent.confirmation_response_queue.get()
            )
            control_task = (
                asyncio.create_task(self.agent.agent_control_queue.get())
                if self.agent.agent_control_queue
                else None
            )

            tasks = [confirmation_task]
            if control_task:
                tasks.append(control_task)

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()

            if confirmation_task in done:
                response_data = confirmation_task.result()
                response = response_data.get("response")
                custom_task = response_data.get("custom_task")

                if response == "yes":
                    return True
                if response == "no":
                    return False
                if custom_task:
                    return custom_task

            if control_task and control_task in done:
                control_data = control_task.result()
                command = control_data.get("command")

                if command == "pause":
                    self.agent.pause()
                    logger.info("Agent paused via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback(
                            {"type": "agent_status", "status": "Paused ⏸️"}
                        )
                elif command == "resume":
                    self.agent.resume()
                    logger.info("Agent resumed via user request during confirmation.")
                    if self.agent.send_agent_message_callback:
                        await self.agent.send_agent_message_callback(
                            {"type": "agent_status", "status": "Participating"}
                        )
                elif command == "stop":
                    self.agent.stop()
                    return False
