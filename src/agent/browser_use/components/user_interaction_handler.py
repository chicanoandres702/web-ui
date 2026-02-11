from __future__ import annotations
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent


class UserInteractionHandler:
    """Handles user interaction requests and decision logic."""

    def __init__(self, agent: BrowserUseAgent):
        self.agent = agent

    async def handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns: True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = (
            self.agent.state.history.history[-1]
            if self.agent.state.history.history
            else None
        )
        intel = "No specific new information from this step."

        if last_step_info and last_step_info.model_output:
            thought = getattr(last_step_info.model_output, "thought", None)
            intel = thought
        if (
            last_step_info
            and last_step_info.model_output
            and last_step_info.model_output.action
        ):
            action_strings = []
            actions = (
                last_step_info.model_output.action
                if isinstance(last_step_info.model_output.action, list)
                else [last_step_info.model_output.action]
            )
            for action_model in actions:
                if hasattr(action_model, "model_dump"):
                    action_dict = action_model.model_dump()
                    action_strings.append(str(action_dict))
                else:
                    action_strings.append(str(action_model))
            next_action_desc = ", ".join(action_strings)
        user_decision = await self.agent.llm_manager.request_user_decision(
            intel, next_action_desc
        )

        if user_decision is False:
            self.agent.state.stopped = True
            return True
        elif isinstance(user_decision, str):
            self.agent.task = user_decision
            self.agent.heuristics.inject_message(
                f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan."
            )
            return False
        return False

    async def request_user_decision(
        self, intel: str, next_task: str
    ) -> Union[bool, str]:
        """
        Internal wrapper for requesting user decision.
        """
        return await self.agent.llm_manager.request_user_decision(intel, next_task)
