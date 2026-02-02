import asyncio
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.tools.file_management import (
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)

# Langchain imports
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool, Tool

# Langgraph imports
from langgraph.graph import StateGraph

from src.utils.browser_factory import create_browser, create_context
from src.browser.custom_browser import CustomBrowser
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.controller.custom_controller import CustomController
from src.utils.prompts import FULL_SYSTEM_PROMPT
from src.utils.mcp_client import setup_mcp_client_and_tools
from src.utils.utils import retry_async
from src.agent.deep_research.types import (
    DeepResearchState, 
    ResearchCategoryItem, 
    ResearchTaskItem, 
    PLAN_FILENAME, 
    SEARCH_INFO_FILENAME, 
    REPORT_FILENAME
)
from src.agent.deep_research.state_manager import (
    load_previous_state, 
    save_plan_to_md, 
    save_search_results_to_json, 
    save_report_to_md, 
    archive_report_to_kb
)
from src.agent.deep_research.search_tool import (
    create_browser_search_tool, 
    create_academic_search_tool,
    create_youtube_search_tool,
    stop_browsers_for_task,
    run_single_browser_task,
    _AGENT_STOP_FLAGS
)
from src.utils.prompts import DEEP_RESEARCH_PLANNING_PROMPT, DEEP_RESEARCH_SYNTHESIS_SYSTEM_PROMPT
from src.agent.deep_research.tool_executor import execute_tools

logger = logging.getLogger(__name__)


# --- Langgraph Nodes ---


async def planning_node(state: DeepResearchState) -> Dict[str, Any]:
    logger.info("--- Entering Planning Node ---")
    if state.get("stop_requested"):
        logger.info("Stop requested, skipping planning.")
        return {"stop_requested": True}

    llm = state["llm"]
    topic = state["topic"]
    existing_plan = state.get("research_plan")
    output_dir = state["output_dir"]

    if existing_plan and (state.get("current_category_index", 0) > 0 or state.get("current_task_index_in_category", 0) > 0):
        logger.info("Resuming with existing plan.")
        await asyncio.to_thread(save_plan_to_md, existing_plan, output_dir)  # Ensure it's saved initially
        # current_category_index and current_task_index_in_category should be set by _load_previous_state
        return {"research_plan": existing_plan}

    logger.info(f"Generating new research plan for topic: {topic}")

    prompt_text = DEEP_RESEARCH_PLANNING_PROMPT.format(topic=topic)
    messages = [
        SystemMessage(content="You are a research planning assistant outputting JSON."),
        HumanMessage(content=prompt_text)
    ]

    try:
        response = await retry_async(llm.ainvoke, messages, logger=logger, error_message="Planning LLM call failed")
        raw_content = response.content
        # The LLM might wrap the JSON in backticks
        if raw_content.strip().startswith("```json"):
            raw_content = raw_content.strip()[7:-3].strip()
        elif raw_content.strip().startswith("```"):
            raw_content = raw_content.strip()[3:-3].strip()

        logger.debug(f"LLM response for plan: {raw_content}")
        parsed_plan_from_llm = json.loads(raw_content)

        new_plan: List[ResearchCategoryItem] = []
        for cat_idx, category_data in enumerate(parsed_plan_from_llm):
            if not isinstance(category_data,
                              dict) or "category_name" not in category_data or "tasks" not in category_data:
                logger.warning(f"Skipping invalid category data: {category_data}")
                continue

            tasks: List[ResearchTaskItem] = []
            for task_idx, task_desc in enumerate(category_data["tasks"]):
                if isinstance(task_desc, str):
                    tasks.append(
                        ResearchTaskItem(
                            task_description=task_desc,
                            status="pending",
                            queries=None,
                            result_summary=None,
                        )
                    )
                else:  # Sometimes LLM puts tasks as {"task": "description"}
                    if isinstance(task_desc, dict) and "task_description" in task_desc:
                        tasks.append(
                            ResearchTaskItem(
                                task_description=task_desc["task_description"],
                                status="pending",
                                queries=None,
                                result_summary=None,
                            )
                        )
                    elif isinstance(task_desc, dict) and "task" in task_desc:  # common LLM mistake
                        tasks.append(
                            ResearchTaskItem(
                                task_description=task_desc["task"],
                                status="pending",
                                queries=None,
                                result_summary=None,
                            )
                        )
                    else:
                        logger.warning(
                            f"Skipping invalid task data: {task_desc} in category {category_data['category_name']}")

            new_plan.append(
                ResearchCategoryItem(
                    category_name=category_data["category_name"],
                    tasks=tasks,
                )
            )

        if not new_plan:
            logger.error("LLM failed to generate a valid plan structure from JSON.")
            return {"error_message": "Failed to generate research plan structure."}

        logger.info(f"Generated research plan with {len(new_plan)} categories.")
        await asyncio.to_thread(save_plan_to_md, new_plan, output_dir, 0, 0)  # Save the hierarchical plan

        return {
            "research_plan": new_plan,
            "current_category_index": 0,
            "current_task_index_in_category": 0,
            "search_results": [],
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM for plan: {e}. Response was: {raw_content}", exc_info=True)
        return {"error_message": f"LLM generated invalid JSON for research plan: {e}"}
    except Exception as e:
        logger.error(f"Error during planning: {e}", exc_info=True)
        return {"error_message": f"LLM Error during planning: {e}"}


async def research_execution_node(state: DeepResearchState) -> Dict[str, Any]:
    logger.info("--- Entering Research Execution Node ---")
    if state.get("stop_requested"):
        logger.info("Stop requested, skipping research execution.")
        return {
            "stop_requested": True,
            "current_category_index": state["current_category_index"],
            "current_task_index_in_category": state["current_task_index_in_category"],
        }

    plan = state["research_plan"]
    cat_idx = state["current_category_index"]
    task_idx = state["current_task_index_in_category"]
    llm = state["llm"]
    tools = state["tools"]
    output_dir = str(state["output_dir"])
    task_id = state["task_id"]  # For _AGENT_STOP_FLAGS

    # This check should ideally be handled by `should_continue`
    if not plan or cat_idx >= len(plan):
        logger.info("Research plan complete or categories exhausted.")
        return {}  # should route to synthesis

    # Update plan file to highlight current task
    await asyncio.to_thread(save_plan_to_md, plan, output_dir, cat_idx, task_idx)

    current_category = plan[cat_idx]
    if task_idx >= len(current_category["tasks"]):
        logger.info(f"All tasks in category '{current_category['category_name']}' completed. Moving to next category.")
        # This logic is now effectively handled by should_continue and the index updates below
        # The next iteration will be caught by should_continue or this node with updated indices
        return {
            "current_category_index": cat_idx + 1,
            "current_task_index_in_category": 0,
            "messages": state["messages"]  # Pass messages along
        }

    current_task = current_category["tasks"][task_idx]

    if current_task["status"] == "completed":
        logger.info(
            f"Task '{current_task['task_description']}' in category '{current_category['category_name']}' already completed. Skipping.")
        # Logic to find next task
        next_task_idx = task_idx + 1
        next_cat_idx = cat_idx
        if next_task_idx >= len(current_category["tasks"]):
            next_cat_idx += 1
            next_task_idx = 0
        return {
            "current_category_index": next_cat_idx,
            "current_task_index_in_category": next_task_idx,
            "messages": state["messages"]  # Pass messages along
        }

    logger.info(
        f"Executing research task: '{current_task['task_description']}' (Category: '{current_category['category_name']}')"
    )

    llm_with_tools = llm.bind_tools(tools)

    # Construct messages for LLM invocation
    task_prompt_content = (
        f"Current Research Category: {current_category['category_name']}\n"
        f"Specific Task: {current_task['task_description']}\n\n"
        "Please use the available tools, especially 'parallel_browser_search', to gather information for this specific task. "
        "Provide focused search queries relevant ONLY to this task. "
        "If you believe you have sufficient information from previous steps for this specific task, you can indicate that you are ready to summarize or that no further search is needed."
    )
    current_task_message_history = [
        HumanMessage(content=task_prompt_content)
    ]
    if not state["messages"]:  # First actual execution message
        invocation_messages = [
                                  SystemMessage(
                                      content="You are a research assistant executing one task of a research plan. Focus on the current task only."),
                              ] + current_task_message_history
    else:
        invocation_messages = state["messages"] + current_task_message_history

    try:
        logger.info(f"Invoking LLM with tools for task: {current_task['task_description']}")
        ai_response: BaseMessage = await retry_async(llm_with_tools.ainvoke, invocation_messages, logger=logger, error_message="Execution LLM call failed")
        logger.info("LLM invocation complete.")

        tool_results = []
        executed_tool_names = []
        current_search_results = state.get("search_results", [])  # Get existing search results

        if not isinstance(ai_response, AIMessage) or not ai_response.tool_calls:
            logger.warning(
                f"LLM did not call any tool for task '{current_task['task_description']}'. Response: {ai_response.content[:100]}..."
            )
            current_task["status"] = "pending"  # Or "completed_no_tool" if LLM explains it's done
            current_task["result_summary"] = f"LLM did not use a tool. Response: {ai_response.content}"
            current_task["current_category_index"] = cat_idx
            current_task["current_task_index_in_category"] = task_idx
            return current_task
            # We still save the plan and advance.
        else:
            # Process tool calls
            tool_results, executed_tool_names, stop_requested = await execute_tools(
                tools, ai_response.tool_calls, task_id, current_search_results
            )

            if stop_requested:
                current_task["status"] = "pending"
                await asyncio.to_thread(save_plan_to_md, plan, output_dir, cat_idx, task_idx)
                return {"stop_requested": True, "research_plan": plan, "current_category_index": cat_idx,
                        "current_task_index_in_category": task_idx}

            # After processing all tool calls for this task
            step_failed_tool_execution = any("Error:" in str(tr.content) for tr in tool_results)
            # Consider a task successful if a browser search was attempted and didn't immediately error out during call
            # The browser search itself returns status for each query. 
            browser_tool_attempted_successfully = "parallel_browser_search" in executed_tool_names and not step_failed_tool_execution

            if step_failed_tool_execution:
                current_task["status"] = "failed"
                current_task[
                    "result_summary"] = f"Tool execution failed. Errors: {[tr.content for tr in tool_results if 'Error' in str(tr.content)]}"
            elif executed_tool_names:  # If any tool was called
                current_task["status"] = "completed"
                current_task["result_summary"] = f"Executed tool(s): {', '.join(executed_tool_names)}."
                # TODO: Could ask LLM to summarize the tool_results for this task if needed, rather than just listing tools.
            else:  # No tool calls but AI response had .tool_calls structure (empty)
                current_task["status"] = "failed"  # Or a more specific status
                current_task["result_summary"] = "LLM prepared for tool call but provided no tools."

        # Save progress
        await asyncio.to_thread(save_plan_to_md, plan, output_dir, next_cat_idx, next_task_idx)
        await asyncio.to_thread(save_search_results_to_json, current_search_results, output_dir)

        # Determine next indices
        next_task_idx = task_idx + 1
        next_cat_idx = cat_idx
        if next_task_idx >= len(current_category["tasks"]):
            next_cat_idx += 1
            next_task_idx = 0

        updated_messages = state["messages"] + current_task_message_history + [ai_response] + tool_results

        return {
            "research_plan": plan,
            "search_results": current_search_results,
            "current_category_index": next_cat_idx,
            "current_task_index_in_category": next_task_idx,
            "messages": updated_messages,
        }

    except Exception as e:
        logger.error(f"Unhandled error during research execution for task '{current_task['task_description']}': {e}",
                     exc_info=True)
        current_task["status"] = "failed"
        await asyncio.to_thread(save_plan_to_md, plan, output_dir, cat_idx, task_idx)
        # Determine next indices even on error to attempt to move on
        next_task_idx = task_idx + 1
        next_cat_idx = cat_idx
        if next_task_idx >= len(current_category["tasks"]):
            next_cat_idx += 1
            next_task_idx = 0
        return {
            "research_plan": plan,
            "current_category_index": next_cat_idx,
            "current_task_index_in_category": next_task_idx,
            "error_message": f"Core Execution Error on task '{current_task['task_description']}': {e}",
            "messages": state["messages"] + current_task_message_history  # Preserve messages up to error
        }


async def synthesis_node(state: DeepResearchState) -> Dict[str, Any]:
    """Synthesizes the final report from the collected search results."""
    logger.info("--- Entering Synthesis Node ---")
    if state.get("stop_requested"):
        logger.info("Stop requested, skipping synthesis.")
        return {"stop_requested": True}

    llm = state["llm"]
    topic = state["topic"]
    search_results = state.get("search_results", [])
    output_dir = state["output_dir"]
    plan = state["research_plan"]  # Include plan for context

    if not search_results:
        logger.warning("No search results found to synthesize report.")
        report = f"# Research Report: {topic}\n\nNo information was gathered during the research process."
        await asyncio.to_thread(save_report_to_md, report, output_dir)
        return {"final_report": report}

    logger.info(
        f"Synthesizing report from {len(search_results)} collected search result entries."
    )

    # Prepare context for the LLM
    # Format search results nicely, maybe group by query or original plan step
    formatted_results = ""
    references = {}
    ref_count = 1
    for i, result_entry in enumerate(search_results):
        query = result_entry.get("query", "Unknown Query")  # From parallel_browser_search
        tool_name = result_entry.get("tool_name")  # From other tools
        status = result_entry.get("status", "unknown")
        result_data = result_entry.get("result")  # From BrowserUseAgent's final_result
        tool_output_str = result_entry.get("output")  # From other tools

        if tool_name in ["parallel_browser_search", "academic_paper_search", "youtube_search"] and status == "completed" and result_data:
            # result_data is the summary from BrowserUseAgent
            if tool_name == "academic_paper_search":
                source_type = "Academic Source"
            elif tool_name == "youtube_search":
                source_type = "YouTube Video"
            else:
                source_type = "Web Search"
                
            formatted_results += f'### Finding from {source_type} Query: "{query}"\n'
            formatted_results += f"- **Summary:**\n{result_data}\n"  # result_data is already a summary string here
            formatted_results += "---\n"
            
        elif tool_name not in ["parallel_browser_search", "academic_paper_search", "youtube_search"] and status == "completed" and tool_output_str:
            formatted_results += f'### Finding from Tool: "{tool_name}" (Args: {result_entry.get("args")})\n'
            formatted_results += f"- **Output:**\n{tool_output_str}\n"
            formatted_results += "---\n"
        elif status == "failed":
            error = result_entry.get("error")
            q_or_t = f"Query: \"{query}\"" if query != "Unknown Query" else f"Tool: \"{tool_name}\""
            formatted_results += f'### Failed {q_or_t}\n'
            formatted_results += f"- **Error:** {error}\n"
            formatted_results += "---\n"

    # Prepare the research plan context
    plan_summary = "\nResearch Plan Followed:\n"
    for cat_idx, category in enumerate(plan):
        plan_summary += f"\n#### Category {cat_idx + 1}: {category['category_name']}\n"
        for task_idx, task in enumerate(category['tasks']):
            marker = "[x]" if task["status"] == "completed" else "[ ]" if task["status"] == "pending" else "[-]"
            plan_summary += f"  - {marker} {task['task_description']}\n"

    synthesis_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system", DEEP_RESEARCH_SYNTHESIS_SYSTEM_PROMPT
            ),
            (
                "human",
                f"""
            **Research Topic:** {topic}

            {plan_summary}

            **Collected Findings:**
            ```
            {formatted_results}
            ```

            Please generate the final research report in Markdown format based **only** on the information above.
            """,
            ),
        ]
    )

    messages = synthesis_prompt.format_prompt(
        topic=topic,
        plan_summary=plan_summary,
        formatted_results=formatted_results,
    ).to_messages()

    try:
        response = await retry_async(llm.ainvoke, messages, logger=logger, error_message="Synthesis LLM call failed")
        final_report_md = response.content

        # Append the reference list automatically to the end of the generated markdown
        if references:
            report_references_section = "\n\n## References\n\n"
            # Sort refs by ID for consistent output
            sorted_refs = sorted(references.values(), key=lambda x: x["id"])
            for ref in sorted_refs:
                report_references_section += (
                    f"[{ref['id']}] {ref['title']} - {ref['url']}\n"
                )
            final_report_md += report_references_section

        logger.info("Successfully synthesized the final report.")
        await asyncio.to_thread(save_report_to_md, final_report_md, output_dir)
        
        if state.get("memory_file"):
            await asyncio.to_thread(archive_report_to_kb, final_report_md, topic, state["memory_file"])
            
        return {"final_report": final_report_md}

    except Exception as e:
        logger.error(f"Error during report synthesis: {e}", exc_info=True)
        return {"error_message": f"LLM Error during synthesis: {e}"}

async def publish_node(state: DeepResearchState) -> Dict[str, Any]:
    """Publishes the final report to Google Docs using a browser agent."""
    logger.info("--- Entering Publish Node ---")
    if state.get("stop_requested"):
        return {"stop_requested": True}
    
    final_report = state.get("final_report")
    if not final_report:
        logger.warning("No final report to publish.")
        return {}

    try:
        logger.info("Attempting to publish report to Google Docs...")
        browser_config = state["browser_config"]
        task_id = state["task_id"]
        
        # Use the template URL from state, defaulting to docs.new
        template_url = state.get("google_docs_template_url") or "https://docs.new"
        
        publish_task = f"""
        Create a new Google Doc based on the assignment requirements and write the research report into it.
        1. Navigate to '{template_url}'.
        2. Wait for the document to load.
        3. Type the content below into the document, maintaining the structure.
        
        REPORT CONTENT:
        {final_report}
        
        5. Name the document based on the report title: "Social Work Theory Analysis - [Character Name]".
        
        6. **INSERT CITATIONS AS FOOTNOTES**:
           - Identify any in-text citations (e.g., (Author, Year)) in the body.
           - For each citation, insert a footnote (Insert > Footnote or Ctrl+Alt+F).
           - In the footnote, paste the full APA citation corresponding to that source from the References list.
        """
        
        result = await run_single_browser_task(
            task_data=publish_task,
            task_id=task_id,
            llm=state["llm"],
            browser_config=browser_config,
            stop_event=_AGENT_STOP_FLAGS.get(task_id),
            use_vision=browser_config.get("use_vision", True),
            memory_file=None,
            shared_browser=None,
            prompt_template=None # Use raw prompt
        )
        
        if result.get("status") == "failed":
             return {"error_message": f"Publishing Error: {result.get('error')}"}
        
        return {"published": True}
        
    except Exception as e:
        logger.error(f"Error publishing to Google Docs: {e}")
        return {"error_message": f"Publishing Error: {e}"}

# --- Langgraph Edges and Conditional Logic ---


def should_continue(state: DeepResearchState) -> str:
    logger.info("--- Evaluating Condition: Should Continue? ---")
    if state.get("stop_requested"):
        logger.info("Stop requested, routing to END.")
        return "end_run"
    if state.get("error_message") and "Core Execution Error" in state["error_message"]:  # Critical error in node
        logger.warning(f"Critical error detected: {state['error_message']}. Routing to END.")
        return "end_run"

    plan = state.get("research_plan")
    cat_idx = state.get("current_category_index", 0)
    task_idx = state.get("current_task_index_in_category", 0)  # This is the *next* task to check

    if not plan:
        logger.warning("No research plan found. Routing to END.")
        return "end_run"

    # Check if the current indices point to a valid pending task
    if cat_idx < len(plan):
        current_category = plan[cat_idx]
        if task_idx < len(current_category["tasks"]):
            # We are trying to execute the task at plan[cat_idx]["tasks"][task_idx]
            # The research_execution_node will handle if it's already completed.
            logger.info(
                f"Plan has potential pending tasks (next up: Category {cat_idx}, Task {task_idx}). Routing to Research Execution."
            )
            return "execute_research"
        else:  # task_idx is out of bounds for current category, means we need to check next category
            if cat_idx + 1 < len(plan):  # If there is a next category
                logger.info(
                    f"Finished tasks in category {cat_idx}. Moving to category {cat_idx + 1}. Routing to Research Execution."
                )
                # research_execution_node will update state to {current_category_index: cat_idx + 1, current_task_index_in_category: 0}
                # Or rather, the previous execution node already set these indices to the start of the next category.
                return "execute_research"

    # If we've gone through all categories and tasks (cat_idx >= len(plan))
    logger.info("All plan categories and tasks processed or current indices are out of bounds. Routing to Synthesis.")
    return "synthesize_report"


# --- DeepSearchAgent Class ---


class DeepResearchAgent:
    def __init__(
            self,
            llm: Any,
            browser_config: Dict[str, Any],
            mcp_server_config: Optional[Dict[str, Any]] = None,
            memory_file: Optional[str] = None,
    ):
        """
        Initializes the DeepSearchAgent.

        Args:
            llm: The Langchain compatible language model instance.
            browser_config: Configuration dictionary for the BrowserUseAgent tool.
                            Example: {"headless": True, "window_width": 1280, ...}
            mcp_server_config: Optional configuration for the MCP client.
            memory_file: Optional path to the memory file for the agent.
        """
        self.llm = llm
        self.browser_config = browser_config
        self.mcp_server_config = mcp_server_config
        self.memory_file = memory_file
        self.mcp_client = None
        self.stopped = False
        self.graph = self._compile_graph()
        self.current_task_id: Optional[str] = None
        self.stop_event: Optional[threading.Event] = None
        self.runner: Optional[asyncio.Task] = None  # To hold the asyncio task for run

    async def _setup_tools(
            self, task_id: str, stop_event: threading.Event, max_parallel_browsers: int = 1,
            shared_browser: Optional[CustomBrowser] = None,
    ) -> List[Tool]:
        """Sets up the basic tools (File I/O) and optional MCP tools."""
        tools = [
            WriteFileTool(),
            ReadFileTool(),
            ListDirectoryTool(),
        ]  # Basic file operations
        browser_use_tool = create_browser_search_tool(
            llm=self.llm,
            browser_config=self.browser_config,
            task_id=task_id,
            stop_event=stop_event,
            max_parallel_browsers=max_parallel_browsers,
            memory_file=self.memory_file,
            shared_browser=shared_browser,
        )
        academic_search_tool = create_academic_search_tool(
            llm=self.llm,
            browser_config=self.browser_config,
            task_id=task_id,
            stop_event=stop_event,
            max_parallel_browsers=max_parallel_browsers,
            memory_file=self.memory_file,
            shared_browser=shared_browser,
        )
        youtube_search_tool = create_youtube_search_tool(
            llm=self.llm,
            browser_config=self.browser_config,
            task_id=task_id,
            stop_event=stop_event,
            max_parallel_browsers=max_parallel_browsers,
            memory_file=self.memory_file,
            shared_browser=shared_browser,
        )
        tools += [browser_use_tool, academic_search_tool, youtube_search_tool]
        # Add MCP tools if config is provided
        if self.mcp_server_config:
            try:
                logger.info("Setting up MCP client and tools...")
                if not self.mcp_client:
                    self.mcp_client = await setup_mcp_client_and_tools(
                        self.mcp_server_config
                    )
                mcp_tools = self.mcp_client.get_tools()
                logger.info(f"Loaded {len(mcp_tools)} MCP tools.")
                tools.extend(mcp_tools)
            except Exception as e:
                logger.error(f"Failed to set up MCP tools: {e}", exc_info=True)
        elif self.mcp_server_config:
            logger.warning(
                "MCP server config provided, but setup function unavailable."
            )
        tools_map = {tool.name: tool for tool in tools}
        return tools_map.values()

    async def close_mcp_client(self):
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            self.mcp_client = None

    def _compile_graph(self) -> StateGraph:
        """Compiles the Langgraph state machine."""
        workflow = StateGraph(DeepResearchState)

        # Add nodes
        workflow.add_node("plan_research", planning_node)
        workflow.add_node("execute_research", research_execution_node)
        workflow.add_node("synthesize_report", synthesis_node)
        workflow.add_node("publish_report", publish_node)
        workflow.add_node(
            "end_run", lambda state: logger.info("--- Reached End Run Node ---") or {}
        )  # Simple end node

        # Define edges
        workflow.set_entry_point("plan_research")

        workflow.add_edge(
            "plan_research", "execute_research"
        )  # Always execute after planning

        # Conditional edge after execution
        workflow.add_conditional_edges(
            "execute_research",
            should_continue,
            {
                "execute_research": "execute_research",  # Loop back if more steps
                "synthesize_report": "synthesize_report",  # Move to synthesis if done
                "end_run": "end_run",  # End if stop requested or error
            },
        )

        workflow.add_edge("synthesize_report", "publish_report")
        workflow.add_edge("publish_report", "end_run")

        app = workflow.compile()
        return app

    async def run(
            self,
            topic: str,
            task_id: Optional[str] = None,
            save_dir: str = "./tmp/deep_research",
            max_parallel_browsers: int = 1,
            google_docs_template_url: str = "https://docs.new",
    ) -> Dict[str, Any]:
        """
        Starts the deep research process (Async Generator Version).

        Args:
            topic: The research topic.
            task_id: Optional existing task ID to resume. If None, a new ID is generated.

        Yields:
             Intermediate state updates or messages during execution.
        """
        if self.runner and not self.runner.done():
            logger.warning(
                "Agent is already running. Please stop the current task first."
            )
            # Return an error status instead of yielding
            return {
                "status": "error",
                "message": "Agent already running.",
                "task_id": self.current_task_id,
            }

        self.current_task_id = task_id if task_id else str(uuid.uuid4())
        safe_root_dir = "./tmp/deep_research"
        normalized_save_dir = os.path.normpath(save_dir)
        if not normalized_save_dir.startswith(os.path.abspath(safe_root_dir)):
            logger.warning(f"Unsafe save_dir detected: {save_dir}. Using default directory.")
            normalized_save_dir = os.path.abspath(safe_root_dir)
        output_dir = os.path.join(normalized_save_dir, self.current_task_id)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(
            f"[AsyncGen] Starting research task ID: {self.current_task_id} for topic: '{topic}'"
        )
        logger.info(f"[AsyncGen] Output directory: {output_dir}")

        self.stop_event = threading.Event()
        _AGENT_STOP_FLAGS[self.current_task_id] = self.stop_event
        
        # Initialize shared browser for this run
        shared_browser = create_browser(self.browser_config)
        
        agent_tools = await self._setup_tools(
            self.current_task_id, self.stop_event, max_parallel_browsers, shared_browser
        ) 
        initial_state: DeepResearchState = {
            "google_docs_template_url": google_docs_template_url,
            "task_id": self.current_task_id,
            "topic": topic,
            "research_plan": [],
            "search_results": [],
            "messages": [],
            "llm": self.llm,
            "tools": agent_tools,
            "output_dir": Path(output_dir),
            "browser_config": self.browser_config,
            "final_report": None,
            "current_category_index": 0,
            "current_task_index_in_category": 0,
            "stop_requested": False,
            "error_message": None,
            "memory_file": self.memory_file,
        }

        if task_id:
            logger.info(f"Attempting to resume task {task_id}...")
            loaded_state = await asyncio.to_thread(load_previous_state, task_id, output_dir)
            initial_state.update(loaded_state)
            if loaded_state.get("research_plan"):
                logger.info(
                    f"Resuming with {len(loaded_state['research_plan'])} plan categories "
                    f"and {len(loaded_state.get('search_results', []))} existing results. "
                    f"Next task: Cat {initial_state['current_category_index']}, Task {initial_state['current_task_index_in_category']}"
                )
                initial_state["topic"] = (
                    topic  # Allow overriding topic even when resuming? Or use stored topic? Let's use new one.
                )
            else:
                logger.warning(
                    f"Resume requested for {task_id}, but no previous plan found. Starting fresh."
                )

        # --- Execute Graph using ainvoke ---
        final_state = None
        status = "unknown"
        message = None
        try:
            logger.info(f"Invoking graph execution for task {self.current_task_id}...")
            self.runner = asyncio.create_task(self.graph.ainvoke(initial_state))
            final_state = await self.runner
            logger.info(f"Graph execution finished for task {self.current_task_id}.")

            # Determine status based on final state
            if self.stop_event and self.stop_event.is_set():
                status = "stopped"
                message = "Research process was stopped by request."
                logger.info(message)
            elif final_state and final_state.get("error_message"):
                status = "error"
                message = final_state["error_message"]
                logger.error(f"Graph execution completed with error: {message}")
            elif final_state and final_state.get("final_report"):
                status = "completed"
                message = "Research process completed successfully."
                logger.info(message)
            else:
                # If it ends without error/report (e.g., empty plan, stopped before synthesis)
                status = "finished_incomplete"
                message = "Research process finished, but may be incomplete (no final report generated)."
                logger.warning(message)

        except asyncio.CancelledError:
            status = "cancelled"
            message = f"Agent run task cancelled for {self.current_task_id}."
            logger.info(message)
            # final_state will remain None or the state before cancellation if checkpointing was used
        except Exception as e:
            status = "error"
            message = f"Unhandled error during graph execution for {self.current_task_id}: {e}"
            logger.error(message, exc_info=True)
            # final_state will remain None or the state before the error
        finally:
            logger.info(f"Cleaning up resources for task {self.current_task_id}")
            task_id_to_clean = self.current_task_id

            self.stop_event = None
            self.current_task_id = None
            self.runner = None  # Mark runner as finished
            if self.mcp_client:
                await self.mcp_client.__aexit__(None, None, None)
            
            if shared_browser:
                await shared_browser.close()

            # Return a result dictionary including the status and the final state if available
            return {
                "status": status,
                "message": message,
                "task_id": task_id_to_clean,  # Use the stored task_id
                "final_state": final_state
                if final_state
                else {},  # Return the final state dict
            }

    async def stop(self):
        """Signals the currently running agent task to stop."""
        if not self.current_task_id or not self.stop_event:
            logger.info("No agent task is currently running.")
            return

        logger.info(f"Stop requested for task ID: {self.current_task_id}")
        self.stop_event.set()  # Signal the stop event
        self.stopped = True
        await stop_browsers_for_task(self.current_task_id)

    def close(self):
        self.stopped = False
