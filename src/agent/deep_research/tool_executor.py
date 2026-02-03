import json
import logging
import asyncio
from functools import partial
from typing import List, Dict, Tuple, Any
from langchain_core.tools import Tool
from langchain_core.messages import ToolMessage
from src.agent.deep_research.search_tool import _AGENT_STOP_FLAGS, SEARCH_TOOL_NAMES
from src.utils.utils import run_tasks_in_parallel

logger = logging.getLogger(__name__)

async def _execute_single_tool(
    tool_call: Dict,
    tools: List[Tool],
    task_id: str
) -> Dict[str, Any]:
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")
    
    result = {
        "tool_message": None,
        "tool_name": tool_name,
        "search_results": [],
        "stop_requested": False,
        "error": None
    }

    selected_tool = next((t for t in tools if t.name == tool_name), None)

    if not selected_tool:
        logger.error(f"LLM called tool '{tool_name}' which is not available.")
        result["tool_message"] = ToolMessage(content=f"Error: Tool '{tool_name}' not found.", tool_call_id=tool_call_id)
        return result

    try:
        stop_event = _AGENT_STOP_FLAGS.get(task_id)
        if stop_event and stop_event.is_set():
            logger.info(f"Stop requested before executing tool: {tool_name}")
            result["stop_requested"] = True
            return result

        logger.info(f"Executing tool: {tool_name}")
        tool_output = await selected_tool.ainvoke(tool_args)
        logger.info(f"Tool '{tool_name}' executed successfully.")

        if tool_name in SEARCH_TOOL_NAMES:
            if isinstance(tool_output, list):
                result["search_results"].extend(tool_output)
        else:
            logger.info(f"Result from tool '{tool_name}': {str(tool_output)[:200]}...")
            result["search_results"].append(
                {"tool_name": tool_name, "args": tool_args, "output": str(tool_output),
                 "status": "completed"})

        result["tool_message"] = ToolMessage(content=json.dumps(tool_output), tool_call_id=tool_call_id)

    except Exception as e:
        logger.error(f"Error executing tool '{tool_name}': {e}", exc_info=True)
        result["tool_message"] = ToolMessage(content=f"Error executing tool {tool_name}: {e}", tool_call_id=tool_call_id)
        result["search_results"].append(
            {"tool_name": tool_name, "args": tool_args, "status": "failed", "error": str(e)})
        result["error"] = str(e)
    
    return result

async def execute_tools(
    tools: List[Tool],
    tool_calls: List[Dict],
    task_id: str,
    current_search_results: List[Dict]
) -> Tuple[List[ToolMessage], List[str], bool]:
    """Executes a list of tool calls concurrently and returns results."""
    
    task_factories = [partial(_execute_single_tool, tc, tools, task_id) for tc in tool_calls]
    results = await run_tasks_in_parallel(task_factories, max_concurrent=10)
    
    tool_results = []
    executed_tool_names = []
    stop_requested = False
    
    for res in results:
        if res.get("stop_requested"):
            stop_requested = True
        
        if res.get("tool_message"):
            tool_results.append(res["tool_message"])
            
        if res.get("tool_name"):
            executed_tool_names.append(res["tool_name"])
            
        if res.get("search_results"):
            current_search_results.extend(res["search_results"])
            
    return tool_results, executed_tool_names, stop_requested