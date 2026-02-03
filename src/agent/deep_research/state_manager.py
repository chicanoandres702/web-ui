import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple, Union
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

class DeepResearchStateManager:
    """Manages state persistence and loading for Deep Research Agent."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = str(output_dir)
        self.plan_file = os.path.join(self.output_dir, PLAN_FILENAME)
        self.search_file = os.path.join(self.output_dir, SEARCH_INFO_FILENAME)
        self.report_file = os.path.join(self.output_dir, REPORT_FILENAME)

    def load_state(self) -> Dict[str, Any]:
        """Loads previous state from files in the output directory."""
        state_updates = {}
        
        # Load Plan
        loaded_plan, next_cat_idx, next_task_idx = self._load_plan()
        if loaded_plan:
            state_updates["research_plan"] = loaded_plan
            state_updates["current_category_index"] = next_cat_idx
            state_updates["current_task_index_in_category"] = next_task_idx
        
        # Load Search Results
        if os.path.exists(self.search_file):
            try:
                with open(self.search_file, "r", encoding="utf-8") as f:
                    state_updates["search_results"] = json.load(f)
                    logger.info(f"Loaded search results from {self.search_file}")
            except Exception as e:
                logger.error(f"Failed to load search results {self.search_file}: {e}")
                state_updates["error_message"] = f"Failed to load search results: {e}"
                
        return state_updates

    def _load_plan(self) -> Tuple[List[ResearchCategoryItem], int, int]:
        """Internal method to parse the plan markdown file."""
        loaded_plan: List[ResearchCategoryItem] = []
        next_cat_idx, next_task_idx = 0, 0
        found_pending = False

        if not os.path.exists(self.plan_file):
            logger.info(f"Plan file {self.plan_file} not found. Will start fresh.")
            return [], 0, 0

        try:
            with open(self.plan_file, "r", encoding="utf-8") as f:
                current_category: Optional[ResearchCategoryItem] = None
                lines = f.readlines()
                cat_counter = 0
                task_counter_in_cat = 0

                for line in lines:
                    line = line.strip()
                    if line.startswith("## "):  # Category
                        if current_category:
                            loaded_plan.append(current_category)
                            if not found_pending:
                                cat_counter += 1
                                task_counter_in_cat = 0
                        category_name = line[line.find(" "):].strip()
                        current_category = ResearchCategoryItem(category_name=category_name, tasks=[])
                    elif (line.startswith("- [ ]") or line.startswith("- [x]") or line.startswith("- [-]")) and current_category:
                        status = "pending"
                        if line.startswith("- [x]"): status = "completed"
                        elif line.startswith("- [-]"): status = "failed"

                        task_desc = line[5:].strip()
                        # Clean up markers added during save
                        if "**(Current)**" in task_desc:
                            task_desc = task_desc.replace("**(Current)**", "").strip()
                        if line.startswith("> "):
                            # If it was marked as current with blockquote
                            task_desc = line[7:].replace("**(Current)**", "").strip()

                        current_category["tasks"].append(
                            ResearchTaskItem(task_description=task_desc, status=status, queries=None, result_summary=None)
                        )
                        
                        if status == "pending" and not found_pending:
                            next_cat_idx = cat_counter
                            next_task_idx = task_counter_in_cat
                            found_pending = True
                        
                        if not found_pending:
                            task_counter_in_cat += 1

                if current_category:
                    loaded_plan.append(current_category)

            if not found_pending and loaded_plan:
                next_cat_idx = len(loaded_plan)
                next_task_idx = 0
                
            logger.info(f"Loaded plan. Next task: Cat {next_cat_idx}, Task {next_task_idx}")
            return loaded_plan, next_cat_idx, next_task_idx

        except Exception as e:
            logger.error(f"Failed to load plan: {e}", exc_info=True)
            return [], 0, 0

    def save_plan(self, plan: List[ResearchCategoryItem], current_cat_idx: int = -1, current_task_idx: int = -1):
        """Saves the research plan to markdown."""
        try:
            content = "# Research Plan\n\n"
            for cat_idx, category in enumerate(plan):
                content += f"## {cat_idx + 1}. {category['category_name']}\n\n"
                for task_idx, task in enumerate(category['tasks']):
                    is_current = (cat_idx == current_cat_idx and task_idx == current_task_idx)
                    
                    marker = "- [ ]"
                    if task["status"] == "completed": marker = "- [x]"
                    elif task["status"] == "failed": marker = "- [-]"
                    
                    line = f"{marker} {task['task_description']}"
                    if is_current:
                        line = f"> {line} **(Current)**"
                    
                    content += f"  {line}\n"
                content += "\n"
            save_text_to_file(self.plan_file, content)
            logger.info(f"Saved plan to {self.plan_file}")
        except Exception as e:
            logger.error(f"Failed to save plan: {e}")

    def save_search_results(self, results: List[Dict[str, Any]]):
        """Saves search results to JSON."""
        try:
            save_text_to_file(self.search_file, json.dumps(results, indent=2, ensure_ascii=False))
            logger.info(f"Saved search results to {self.search_file}")
        except Exception as e:
            logger.error(f"Failed to save search results: {e}")

    def save_report(self, report: str):
        """Saves the final report."""
        try:
            save_text_to_file(self.report_file, report)
            logger.info(f"Saved report to {self.report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def archive_report(self, report: str, topic: str, memory_file: str):
        """Archives report to knowledge base."""
        try:
            filepath = save_to_knowledge_base_file(report, f"Research Report: {topic}", memory_file)
            if filepath:
                logger.info(f"Archived report to KB: {filepath}")
        except Exception as e:
            logger.error(f"Failed to archive report: {e}")

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

def parse_research_plan(parsed_plan_from_llm: Any) -> List[ResearchCategoryItem]:
    """Parses the JSON output from the LLM into a structured research plan."""
    new_plan: List[ResearchCategoryItem] = []
    
    if not isinstance(parsed_plan_from_llm, list):
        logger.warning(f"Expected list for plan, got {type(parsed_plan_from_llm)}")
        return []

    for category_data in parsed_plan_from_llm:
        if not isinstance(category_data, dict) or "category_name" not in category_data or "tasks" not in category_data:
            logger.warning(f"Skipping invalid category data: {category_data}")
            continue

        tasks: List[ResearchTaskItem] = []
        for task_desc in category_data["tasks"]:
            if isinstance(task_desc, str):
                tasks.append(
                    ResearchTaskItem(
                        task_description=task_desc,
                        status="pending",
                        queries=None,
                        result_summary=None,
                    )
                )
            elif isinstance(task_desc, dict):
                desc = task_desc.get("task_description") or task_desc.get("task")
                if desc:
                    tasks.append(
                        ResearchTaskItem(
                            task_description=desc,
                            status="pending",
                            queries=None,
                            result_summary=None,
                        )
                    )

        new_plan.append(
            ResearchCategoryItem(
                category_name=category_data["category_name"],
                tasks=tasks,
            )
        )
    return new_plan