import logging
from typing import List, Tuple
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)

def assess_task_completion(tool_results: List[ToolMessage], executed_tool_names: List[str]) -> Tuple[str, str]:
    """
    Assess the completion status of a task based on tool execution results.
    Returns: (status, result_summary)
    """
    step_failed_tool_execution = any("Error:" in str(tr.content) for tr in tool_results)
    
    if step_failed_tool_execution:
        errors = [tr.content for tr in tool_results if 'Error' in str(tr.content)]
        return "failed", f"Tool execution failed. Errors: {errors}"
    
    if executed_tool_names:  # If any tool was called
        return "completed", f"Executed tool(s): {', '.join(executed_tool_names)}."
    
    # No tool calls but AI response had .tool_calls structure (empty)
    return "failed", "LLM prepared for tool call but provided no tools."