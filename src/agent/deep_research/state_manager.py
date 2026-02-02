import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.utils.utils import save_text_to_file, save_to_knowledge_base_file
from src.agent.deep_research.types import (
    ResearchCategoryItem, 
    ResearchTaskItem, 
    PLAN_FILENAME, 
    SEARCH_INFO_FILENAME, 
    REPORT_FILENAME
)

logger = logging.getLogger(__name__)

def load_previous_state(task_id: str, output_dir: str) -> Dict[str, Any]:
    state_updates = {}
    plan_file = os.path.join(output_dir, PLAN_FILENAME)
    search_file = os.path.join(output_dir, SEARCH_INFO_FILENAME)

    loaded_plan: List[ResearchCategoryItem] = []
    next_cat_idx, next_task_idx = 0, 0
    found_pending = False

    if os.path.exists(plan_file):
        try:
            with open(plan_file, "r", encoding="utf-8") as f:
                current_category: Optional[ResearchCategoryItem] = None
                lines = f.readlines()
                cat_counter = 0
                task_counter_in_cat = 0

                for line_num, line_content in enumerate(lines):
                    line = line_content.strip()
                    if line.startswith("## "):  # Category
                        if current_category:  # Save previous category
                            loaded_plan.append(current_category)
                            if not found_pending:  # If previous category was all done, advance cat counter
                                cat_counter += 1
                                task_counter_in_cat = 0
                        category_name = line[line.find(" "):].strip()  # Get text after "## X. "
                        current_category = ResearchCategoryItem(category_name=category_name, tasks=[])
                    elif (line.startswith("- [ ]") or line.startswith("- [x]") or line.startswith(
                            "- [-]")) and current_category:  # Task
                        status = "pending"
                        if line.startswith("- [x]"):
                            status = "completed"
                        elif line.startswith("- [-]"):
                            status = "failed"

                        task_desc = line[5:].strip()
                        current_category["tasks"].append(
                            ResearchTaskItem(task_description=task_desc, status=status, queries=None,
                                             result_summary=None)
                        )
                        if status == "pending" and not found_pending:
                            next_cat_idx = cat_counter
                            next_task_idx = task_counter_in_cat
                            found_pending = True
                        if not found_pending:  # only increment if previous tasks were completed/failed
                            task_counter_in_cat += 1

                if current_category:  # Append last category
                    loaded_plan.append(current_category)

            if loaded_plan:
                state_updates["research_plan"] = loaded_plan
                if not found_pending and loaded_plan:  # All tasks were completed or failed
                    next_cat_idx = len(loaded_plan)  # Points beyond the last category
                    next_task_idx = 0
                state_updates["current_category_index"] = next_cat_idx
                state_updates["current_task_index_in_category"] = next_task_idx
                logger.info(
                    f"Loaded hierarchical research plan from {plan_file}. "
                    f"Next task: Category {next_cat_idx}, Task {next_task_idx} in category."
                )
            else:
                logger.warning(f"Plan file {plan_file} was empty or malformed.")

        except Exception as e:
            logger.error(f"Failed to load or parse research plan {plan_file}: {e}", exc_info=True)
            state_updates["error_message"] = f"Failed to load research plan: {e}"
    else:
        logger.info(f"Plan file {plan_file} not found. Will start fresh.")

    if os.path.exists(search_file):
        try:
            with open(search_file, "r", encoding="utf-8") as f:
                state_updates["search_results"] = json.load(f)
                logger.info(f"Loaded search results from {search_file}")
        except Exception as e:
            logger.error(f"Failed to load search results {search_file}: {e}")
            state_updates["error_message"] = (
                    state_updates.get("error_message", "") + f" Failed to load search results: {e}").strip()

    return state_updates


def save_plan_to_md(plan: List[ResearchCategoryItem], output_dir: str, current_cat_idx: int = -1, current_task_idx: int = -1):
    plan_file = os.path.join(output_dir, PLAN_FILENAME)
    try:
        content = f"# Research Plan\n\n"
        for cat_idx, category in enumerate(plan):
            content += f"## {cat_idx + 1}. {category['category_name']}\n\n"
            for task_idx, task in enumerate(category['tasks']):
                is_current = (cat_idx == current_cat_idx and task_idx == current_task_idx)
                
                if task["status"] == "completed":
                    marker = "- [x]"
                elif task["status"] == "failed":
                    marker = "- [-]"
                else:
                    marker = "- [ ]"
                
                if is_current:
                    marker = f"> {marker} **(Current)**"
                
                content += f"  {marker} {task['task_description']}\n"
            content += "\n"
        save_text_to_file(plan_file, content)
        logger.info(f"Hierarchical research plan saved to {plan_file}")
    except Exception as e:
        logger.error(f"Failed to save research plan to {plan_file}: {e}")


def save_search_results_to_json(results: List[Dict[str, Any]], output_dir: str):
    """Appends or overwrites search results to a JSON file."""
    search_file = os.path.join(output_dir, SEARCH_INFO_FILENAME)
    try:
        # Simple overwrite for now, could be append
        save_text_to_file(search_file, json.dumps(results, indent=2, ensure_ascii=False))
        logger.info(f"Search results saved to {search_file}")
    except Exception as e:
        logger.error(f"Failed to save search results to {search_file}: {e}")

def save_report_to_md(report: str, output_dir: Path):
    """Saves the final report to a markdown file."""
    report_file = os.path.join(output_dir, REPORT_FILENAME)
    try:
        save_text_to_file(report_file, report)
        logger.info(f"Final report saved to {report_file}")
    except Exception as e:
        logger.error(f"Failed to save final report to {report_file}: {e}")

def archive_report_to_kb(report: str, topic: str, memory_file: str):
    """Archives the research report to the knowledge base."""
    try:
        filepath = save_to_knowledge_base_file(report, f"Research Report: {topic}", memory_file)
        if not filepath: return
        logger.info(f"Archived research report to knowledge base: {filepath}")
    except Exception as e:
        logger.error(f"Failed to archive report to knowledge base: {e}")


def get_next_task_indices(plan: List[ResearchCategoryItem], current_cat_idx: int, current_task_idx: int) -> Tuple[int, int]:
    """Calculates the indices for the next task in the plan."""
    next_task_idx = current_task_idx + 1
    next_cat_idx = current_cat_idx
    
    if next_cat_idx < len(plan):
        current_category = plan[next_cat_idx]
        if next_task_idx >= len(current_category["tasks"]):
            next_cat_idx += 1
            next_task_idx = 0
            
    return next_cat_idx, next_task_idx