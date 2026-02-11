import asyncio
import logging
from typing import Any, Union

logger = logging.getLogger(__name__)

class UserInteractionHandler:
    """Handles user interaction requests for the agent."""

    def __init__(self, agent: Any):
        """Initializes the UserInteractionHandler with the agent instance."""
        self.agent = agent

    async def handle_user_interaction(self) -> bool:
        """
        Handles the user interaction dialog.
        Returns True if the agent should stop (User aborted), False otherwise.
        """
        if not self.agent.enable_user_interaction_dialog:
            return False

        last_step_info = self.agent.state.history.history[-1] if self.agent.state.history.history else None
        intel = "No specific new information from this step."
        if last_step_info and last_step_info.model_output:
            thought = getattr(last_step_info.model_output, "thought", None) # type: ignore
            if thought:
                intel = str(thought) # type: ignore

        next_action_desc = "Agent is considering its next move."
        if (last_step_info and last_step_info.model_output and
                last_step_info.model_output.action):
            action_strings = []
            actions = last_step_info.model_output.action if isinstance(last_step_info.model_output.action, list) else [last_step_info.model_output.action]
            for action_model in actions:
                if hasattr(action_model, "model_dump"):
                    action_dict = action_model.model_dump()
                    action_strings.append(str(action_dict))
                else:
                    action_strings.append(str(action_model))  # Fallback for non-standard actions
            
            if action_strings:
                next_action_desc = ", ".join(action_strings)
        if not next_action_desc:
            next_action_desc = "Agent is considering its next move."
                        

        user_decision = await self._request_user_decision(intel, next_action_desc)

        if user_decision is False:  # User chose No/Abort
            self.agent.state.stopped = True
            return True  # Break the loop
        elif isinstance(user_decision, str):  # User provided a custom task
            self.agent.task = user_decision  # Update the main task for the agent to re-evaluate

            self.agent.heuristics.inject_message(f"SYSTEM: User provided new task: '{user_decision}'. Re-evaluating plan.")

            return False  # Continue the loop, agent will now work on the new task
        return False

    async def _request_user_decision(self, intel: str, next_task: str) -> Union[bool, str]:
        """
        Requests confirmation/input from the user via WebSocket.
        """
        if not self.agent.send_agent_message_callback or not self.agent.confirmation_response_queue:
            return True

        await self.agent.send_agent_message_callback({
            "type": "request_confirmation",
            "message": "Agent is pausing for your review.",
            "intel": intel,
            "next_task": next_task
        })

        response_data = await self.agent.confirmation_response_queue.get()
        response = response_data.get("response")
        custom_task = response_data.get("custom_task")

        if response == "yes": return True
        if response == "no": return False
        if custom_task: return custom_task
        return False