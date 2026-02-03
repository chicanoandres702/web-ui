import logging
from typing import List, Optional, Tuple, Dict, Any
from src.agent.deep_research.types import ResearchCategoryItem, ResearchTaskItem

logger = logging.getLogger(__name__)

class TaskManager:
    """
    Manages the hierarchical research plan (Categories -> Tasks).
    Provides methods for navigation, status updates, and dynamic plan modification.
    """
    def __init__(self, plan: List[ResearchCategoryItem]):
        self.plan = plan

    @property
    def total_tasks(self) -> int:
        return sum(len(cat["tasks"]) for cat in self.plan)

    def get_task(self, cat_idx: int, task_idx: int) -> Optional[ResearchTaskItem]:
        if 0 <= cat_idx < len(self.plan):
            tasks = self.plan[cat_idx]["tasks"]
            if 0 <= task_idx < len(tasks):
                return tasks[task_idx]
        return None

    def get_next_task_indices(self, current_cat_idx: int, current_task_idx: int) -> Tuple[int, int]:
        """
        Calculates the indices for the next task in the plan.
        """
        next_task_idx = current_task_idx + 1
        next_cat_idx = current_cat_idx
        
        if next_cat_idx < len(self.plan):
            current_category = self.plan[next_cat_idx]
            if next_task_idx >= len(current_category["tasks"]):
                next_cat_idx += 1
                next_task_idx = 0
                
        return next_cat_idx, next_task_idx

    def get_next_pending_task(self, current_cat_idx: int, current_task_idx: int) -> Tuple[int, int]:
        """
        Finds the next pending task, skipping completed or failed ones.
        Returns (cat_idx, task_idx) or (-1, -1) if finished.
        """
        # Check current task first
        task = self.get_task(current_cat_idx, current_task_idx)
        if task and task["status"] == "pending":
            return current_cat_idx, current_task_idx

        # Search forward
        cat_idx, task_idx = current_cat_idx, current_task_idx
        while cat_idx < len(self.plan):
            tasks = self.plan[cat_idx]["tasks"]
            # If we are in the current category, start from next task
            start_task = task_idx + 1 if cat_idx == current_cat_idx else 0
            
            for i in range(start_task, len(tasks)):
                if tasks[i]["status"] == "pending":
                    return cat_idx, i
            
            cat_idx += 1
            task_idx = -1 # Reset for next category
            
        return -1, -1

    def add_task(self, task_description: str, category_index: int, position: str = "end") -> None:
        """
        Adds a task to a specific category.
        """
        if 0 <= category_index < len(self.plan):
            new_task: ResearchTaskItem = {
                "task_description": task_description,
                "status": "pending",
                "queries": [],
                "result_summary": None
            }
            self.plan[category_index]["tasks"].append(new_task)
        else:
            logger.warning(f"Category index {category_index} out of bounds. Creating new category.")
            self.plan.append({
                "category_name": "Additional Research",
                "tasks": [{
                    "task_description": task_description,
                    "status": "pending",
                    "queries": [],
                    "result_summary": None
                }]
            })

    def update_task_status(self, cat_idx: int, task_idx: int, status: str, result: str = None):
        task = self.get_task(cat_idx, task_idx)
        if task:
            task["status"] = status
            if result:
                task["result_summary"] = result

    def merge_plan(self, new_plan: List[ResearchCategoryItem]):
        """
        Appends new categories from a renegotiated plan to the existing plan.
        """
        self.plan.extend(new_plan)

    def add_subtasks(self, cat_idx: int, task_idx: int, subtasks: List[str]) -> None:
        """
        Injects new tasks immediately after the current task in the same category.
        Useful for breaking down a complex task based on site assessment.
        """
        if not (0 <= cat_idx < len(self.plan)):
            logger.warning(f"Invalid category index {cat_idx} for adding subtasks.")
            return

        new_items: List[ResearchTaskItem] = [
            {
                "task_description": desc,
                "status": "pending",
                "queries": [],
                "result_summary": None
            }
            for desc in subtasks
        ]
        
        # Insert after current task
        insert_pos = task_idx + 1
        self.plan[cat_idx]["tasks"][insert_pos:insert_pos] = new_items
        logger.info(f"Added {len(subtasks)} subtasks to Category {cat_idx} after Task {task_idx}")

    def renegotiate_plan(self, current_cat_idx: int, current_task_idx: int, new_plan: List[ResearchCategoryItem]):
        """
        Dynamically updates the plan starting from the current position.
        Keeps completed tasks, removes pending future tasks, and appends the new plan.
        """
        # 1. Prune future tasks in current category
        if 0 <= current_cat_idx < len(self.plan):
            # Keep tasks up to current_task_idx (inclusive)
            self.plan[current_cat_idx]["tasks"] = self.plan[current_cat_idx]["tasks"][:current_task_idx + 1]
        
        # 2. Remove future categories
        self.plan = self.plan[:current_cat_idx + 1]
        
        # 3. Merge or Append new plan
        if not new_plan:
            return

        # Check if first new category matches current category name for seamless merging
        if self.plan and new_plan[0]["category_name"] == self.plan[-1]["category_name"]:
            self.plan[-1]["tasks"].extend(new_plan[0]["tasks"])
            self.plan.extend(new_plan[1:])
        else:
            self.plan.extend(new_plan)
        
        logger.info(f"Renegotiated plan. New total categories: {len(self.plan)}")

    def get_plan_summary(self) -> str:
        """Returns a string representation of the current plan status."""
        summary = []
        for i, cat in enumerate(self.plan):
            summary.append(f"Category {i+1}: {cat['category_name']}")
            for j, task in enumerate(cat['tasks']):
                status_icon = "✅" if task['status'] == 'completed' else "❌" if task['status'] == 'failed' else "⏳"
                summary.append(f"  {j+1}. {status_icon} {task['task_description']}")
        return "\n".join(summary)

    @staticmethod
    def parse_plan_json(parsed_plan_from_llm: Any) -> List[ResearchCategoryItem]:
        """Parses the JSON output from the LLM into a structured research plan."""
        new_plan: List[ResearchCategoryItem] = []
        
        if not isinstance(parsed_plan_from_llm, list):
            logger.warning(f"Expected list for plan, got {type(parsed_plan_from_llm)}")
            return []

        for category_data in parsed_plan_from_llm:
            if not isinstance(category_data, dict) or "category_name" not in category_data or "tasks" not in category_data:
                continue

            tasks: List[ResearchTaskItem] = []
            for task_desc in category_data["tasks"]:
                if isinstance(task_desc, str):
                    tasks.append({
                        "task_description": task_desc,
                        "status": "pending",
                        "queries": None,
                        "result_summary": None,
                    })
                elif isinstance(task_desc, dict):
                    desc = task_desc.get("task_description") or task_desc.get("task")
                    if desc:
                        tasks.append({
                            "task_description": desc,
                            "status": "pending",
                            "queries": None,
                            "result_summary": None,
                        })

            new_plan.append({
                "category_name": category_data["category_name"],
                "tasks": tasks,
            })
        return new_plan
