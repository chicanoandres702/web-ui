import asyncio
import logging
import threading
import uuid
from typing import Any, Dict, List, Optional
from functools import partial

from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils.browser_factory import create_browser, create_context
from src.browser.custom_browser import CustomBrowser
from src.controller.custom_controller import CustomController
from src.utils.prompts import FULL_SYSTEM_PROMPT, DEEP_RESEARCH_BROWSER_TASK_PROMPT, DEEP_RESEARCH_ACADEMIC_SEARCH_PROMPT, DEEP_RESEARCH_YOUTUBE_SEARCH_PROMPT
from src.utils.utils import run_tasks_in_parallel
logger = logging.getLogger(__name__)

_AGENT_STOP_FLAGS = {}
_BROWSER_AGENT_INSTANCES = {}

SEARCH_TOOL_DEFINITIONS = [
    {
        "name": "parallel_browser_search",
        "description": "Use this tool to actively search the web for information related to a specific research task or question. It runs up to {max_parallel_browsers} searches in parallel using a browser agent for better results than simple scraping. Provide a list of distinct search queries(up to {max_parallel_browsers}) that are likely to yield relevant information.",
        "prompt_template": DEEP_RESEARCH_BROWSER_TASK_PROMPT,
    },
    {
        "name": "academic_paper_search",
        "description": "Use this tool to search specifically for academic papers, journals, and technical reports. It runs up to {max_parallel_browsers} searches in parallel. Provide a list of distinct search queries focused on finding scholarly sources and extracting APA citations.",
        "prompt_template": DEEP_RESEARCH_ACADEMIC_SEARCH_PROMPT,
    },
    {
        "name": "youtube_search",
        "description": "Use this tool to search for YouTube videos and extract transcripts/summaries. It runs up to {max_parallel_browsers} searches in parallel. Provide a list of distinct search queries.",
        "prompt_template": DEEP_RESEARCH_YOUTUBE_SEARCH_PROMPT,
    },
]
SEARCH_TOOL_NAMES = [d["name"] for d in SEARCH_TOOL_DEFINITIONS]

class BrowserSearchInput(BaseModel):
    queries: List[str] = Field(
        description="List of distinct search queries to find information relevant to the research task."
    )

async def run_single_browser_task(
        task_data: str,
        task_id: str,
        llm: Any,  # Pass the main LLM
        browser_config: Dict[str, Any],
        stop_event: Optional[threading.Event] = None,
        use_vision: bool = False,
        memory_file: Optional[str] = None,
        shared_browser: Optional[CustomBrowser] = None,
        prompt_template: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Runs a single BrowserUseAgent task.
    Manages browser creation and closing for this specific task.
    """
    if not BrowserUseAgent:
        return {
            "query": task_data,
            "error": "BrowserUseAgent components not available.",
        }

    # --- Browser Setup ---
    bu_browser = None
    bu_browser_context = None
    own_browser = False
    try:
        logger.info(f"Starting browser task for: {task_data[:50]}...")
        
        if shared_browser:
            bu_browser = shared_browser
        else:
            bu_browser = create_browser(browser_config)
            own_browser = True
        
        # Ensure downloads path is set for context and is task-specific
        context_settings = browser_config.copy()
        # Use a task-specific download directory to keep things organized
        download_path = os.path.join("./tmp/deep_research", task_id, "downloads")
        context_settings["save_downloads_path"] = download_path
            
        bu_browser_context = await create_context(bu_browser, context_settings)

        # Simple controller example, replace with your actual implementation if needed
        bu_controller = CustomController()
        if memory_file:
            bu_controller.set_memory_file(memory_file)

        # Construct the task prompt for BrowserUseAgent
        # Instruct it to find specific info and return title/URL
        if prompt_template:
            bu_task_prompt = prompt_template.format(task_query=task_data)
        else:
            # If no template is provided, treat task_data as the full prompt
            bu_task_prompt = task_data

        bu_agent_instance = BrowserUseAgent(
            task=bu_task_prompt,
            llm=llm,  # Use the passed LLM
            browser=bu_browser,
            browser_context=bu_browser_context,
            controller=bu_controller,
            use_vision=use_vision,
            source="webui",
            extend_system_message=FULL_SYSTEM_PROMPT,
        )

        # Store instance for potential stop() call
        task_key = f"{task_id}_{uuid.uuid4()}"
        _BROWSER_AGENT_INSTANCES[task_key] = bu_agent_instance

        # --- Run with Stop Check ---
        if stop_event and stop_event.is_set():
            logger.info(f"Browser task for '{task_data}' cancelled before start.")
            return {"query": task_data, "result": None, "status": "cancelled"}

        logger.info(f"Running BrowserUseAgent for: {task_data[:50]}...")
        try:
            # Use a reasonable max_steps limit (e.g., 20) and timeout to prevent getting stuck
            result = await asyncio.wait_for(bu_agent_instance.run(max_steps=20), timeout=120)
        except asyncio.TimeoutError:
            logger.warning(f"Browser task timed out for: {task_data}")
            return {"query": task_data, "error": "Task timed out", "status": "failed"}
        logger.info(f"BrowserUseAgent finished for: {task_data[:50]}...")

        final_data = result.final_result()

        if stop_event and stop_event.is_set():
            logger.info(f"Browser task for '{task_data}' stopped during execution.")
            return {"query": task_data, "result": final_data, "status": "stopped"}
        else:
            logger.info(f"Browser result for '{task_data[:50]}...': {final_data}")
            return {"query": task_data, "result": final_data, "status": "completed"}

    except Exception as e:
        logger.error(
            f"Error during browser task for '{task_data}': {e}", exc_info=True
        )
        return {"query": task_query, "error": str(e), "status": "failed"}
    finally:
        if bu_browser_context:
            try:
                await bu_browser_context.close()
                bu_browser_context = None
                logger.info("Closed browser context.")
            except Exception as e:
                logger.error(f"Error closing browser context: {e}")
        if own_browser and bu_browser:
            try:
                await bu_browser.close()
                bu_browser = None
                logger.info("Closed browser.")
            except Exception as e:
                logger.error(f"Error closing browser: {e}")

        if task_key in _BROWSER_AGENT_INSTANCES:
            del _BROWSER_AGENT_INSTANCES[task_key]


async def _run_browser_search_tool(
        queries: List[str],
        task_id: str,
        llm: Any,
        browser_config: Dict[str, Any],
        stop_event: threading.Event,
        max_parallel_browsers: int = 1,
        memory_file: Optional[str] = None,
        shared_browser: Optional[CustomBrowser] = None,
        prompt_template: Optional[str] = None,
        tool_name: str = "parallel_browser_search",
) -> List[Dict[str, Any]]:
    """
    Internal function to execute parallel browser searches based on LLM-provided queries.
    Handles concurrency and stop signals.
    """
    logger.info(
        f"[Browser Tool {task_id}] Running search for {len(queries)} queries: {queries}"
    )

    async def task_wrapper(query):
        if stop_event.is_set():
            logger.info(f"[Browser Tool {task_id}] Skipping task due to stop signal: {query}")
            return {"query": query, "result": None, "status": "cancelled"}
        return await run_single_browser_task(
            query, task_id, llm, browser_config, stop_event,
            memory_file=memory_file, shared_browser=shared_browser, prompt_template=prompt_template
        )

    task_factories = [partial(task_wrapper, query) for query in queries]
    search_results = await run_tasks_in_parallel(task_factories, max_concurrent=max_parallel_browsers)

    processed_results = []
    for i, res in enumerate(search_results):
        query = queries[i]
        if isinstance(res, Exception):
            logger.error(f"[Browser Tool {task_id}] Gather caught exception for query '{query}': {res}", exc_info=True)
            processed_results.append({"query": query, "error": str(res), "status": "failed"})
            # Ensure tool_name is preserved if possible, though here we are inside the function
        elif isinstance(res, dict):
            processed_results.append(res)
        else:
            logger.error(f"[Browser Tool {task_id}] Unexpected result type for query '{query}': {type(res)}")
            processed_results.append({"query": query, "error": "Unexpected result type", "status": "failed"})

    logger.info(f"[Browser Tool {task_id}] Finished search. Results count: {len(processed_results)}")
    
    # Inject tool name into results for synthesis context
    for res in processed_results:
        res["tool_name"] = tool_name
        
    return processed_results


def _create_generic_search_tool(
        name: str,
        description: str,
        prompt_template: str,
        llm: Any,
        browser_config: Dict[str, Any],
        task_id: str,
        stop_event: threading.Event,
        max_parallel_browsers: int = 1,
        memory_file: Optional[str] = None,
        shared_browser: Optional[CustomBrowser] = None,
) -> StructuredTool:
    bound_tool_func = partial(_run_browser_search_tool, task_id=task_id, llm=llm, browser_config=browser_config, stop_event=stop_event, max_parallel_browsers=max_parallel_browsers, memory_file=memory_file, shared_browser=shared_browser, prompt_template=prompt_template, tool_name=name)
    return StructuredTool.from_function(
        coroutine=bound_tool_func,
        name=name,
        description=description,
        args_schema=BrowserSearchInput,
    )

def create_all_search_tools(
        llm: Any,
        browser_config: Dict[str, Any],
        task_id: str,
        stop_event: threading.Event,
        max_parallel_browsers: int = 1,
        memory_file: Optional[str] = None,
        shared_browser: Optional[CustomBrowser] = None,
) -> List[StructuredTool]:
    """Factory function to create all search tools."""
    tools = []
    for tool_def in SEARCH_TOOL_DEFINITIONS:
        description = tool_def["description"].format(max_parallel_browsers=max_parallel_browsers)
        tools.append(_create_generic_search_tool(
            name=tool_def["name"],
            description=description,
            prompt_template=tool_def["prompt_template"],
            llm=llm,
            browser_config=browser_config,
            task_id=task_id,
            stop_event=stop_event,
            max_parallel_browsers=max_parallel_browsers,
            memory_file=memory_file,
            shared_browser=shared_browser,
        ))
    return tools

async def stop_browsers_for_task(task_id: str):
    """Attempts to stop any BrowserUseAgent instances associated with the task_id."""
    keys_to_stop = [key for key in _BROWSER_AGENT_INSTANCES if key.startswith(f"{task_id}_")]
    if not keys_to_stop:
        return

    logger.warning(f"Found {len(keys_to_stop)} potentially lingering browser agents for task {task_id}. Attempting stop...")
    for key in keys_to_stop:
        agent_instance = _BROWSER_AGENT_INSTANCES.get(key)
        try:
            if agent_instance:
                await agent_instance.stop()
                logger.info(f"Called stop() on browser agent instance {key}")
        except Exception as e:
            logger.error(f"Error calling stop() on browser agent instance {key}: {e}")