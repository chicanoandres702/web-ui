import json
import os
import logging
from typing import List, Optional, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.utils.io_manager import IOManager
from src.utils.utils import parse_json_safe
from src.utils.prompts import STREAMLINE_PLAN_PROMPT

logger = logging.getLogger(__name__)

class WorkflowManager:
    def __init__(self, workflows_dir: str = "./tmp/workflows"):
        self.workflows_dir = workflows_dir
        os.makedirs(self.workflows_dir, exist_ok=True)

    def list_workflows(self) -> List[str]:
        if not os.path.exists(self.workflows_dir): return []
        workflows = []
        for root, _, files in os.walk(self.workflows_dir):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.workflows_dir)
                    # Normalize to forward slashes for UI consistency
                    name = os.path.splitext(rel_path)[0].replace(os.sep, "/")
                    workflows.append(name)
        return sorted(workflows)

    def save_workflow(self, name: str, plan: List[dict]) -> bool:
        if not name or not plan: return False
        # Normalize input name to OS path separators
        safe_name = name.replace("/", os.sep)
        filepath = os.path.join(self.workflows_dir, f"{safe_name}.json")
        try:
            # Save full plan details, stripping status, and adding summary
            clean_plan = []
            for step in plan:
                item = {"description": step.get("step", "")}
                if "action" in step: item["action"] = step["action"]
                if "params" in step: item["params"] = step["params"]
                clean_plan.append(item)
            
            # Generate summary if not present
            summary = f"Workflow '{name}' with {len(clean_plan)} steps."
            if len(clean_plan) > 0:
                summary += f" Starts with: {clean_plan[0]['description'][:50]}..."
            
            workflow_data = {
                "name": name,
                "summary": summary,
                "steps": clean_plan,
                "created_at": None # Could add timestamp here
            }
            
            # IOManager handles directory creation
            return IOManager.write_file_sync(filepath, json.dumps(workflow_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save workflow: {e}")
            return False

    def load_workflow(self, name: str) -> List[dict]:
        safe_name = name.replace("/", os.sep)
        filepath = os.path.join(self.workflows_dir, f"{safe_name}.json")
        content = IOManager.read_file_sync(filepath)
        if not content: return []
        try:
            steps = json.loads(content)
            
            # Handle legacy format (list of steps) vs new format (dict with metadata)
            if isinstance(steps, dict) and "steps" in steps:
                steps = steps["steps"]
            elif isinstance(steps, list):
                pass # Legacy format
                
            # Reset status to pending for new execution
            loaded_plan = []
            for s in steps:
                if isinstance(s, str):
                    loaded_plan.append({"step": s, "status": "pending"})
                elif isinstance(s, dict):
                    item = {"step": s.get("description", s.get("step", "")), "status": "pending"}
                    if "action" in s: item["action"] = s["action"]
                    if "params" in s: item["params"] = s["params"]
                    loaded_plan.append(item)
            return loaded_plan
        except Exception as e:
            logger.error(f"Failed to load workflow: {e}")
            return []

    def delete_workflow(self, name: str) -> bool:
        if not name: return False
        safe_name = name.replace("/", os.sep)
        filepath = os.path.join(self.workflows_dir, f"{safe_name}.json")
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                
                # Cleanup empty subdirectories
                directory = os.path.dirname(filepath)
                root_abs = os.path.abspath(self.workflows_dir)
                dir_abs = os.path.abspath(directory)
                
                # Walk up and remove empty dirs until root
                while dir_abs.startswith(root_abs) and dir_abs != root_abs:
                    try:
                        if not os.listdir(dir_abs):
                            os.rmdir(dir_abs)
                            dir_abs = os.path.dirname(dir_abs)
                        else:
                            break
                    except OSError:
                        break
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete workflow: {e}")
            return False

    async def streamline_plan(self, llm: Any, current_plan: List[dict], history_str: str = "") -> List[dict]:
        """
        Uses LLM to optimize the current plan, optionally using execution history.
        """
        steps = []
        for step in current_plan:
            s = {"description": step.get("step", "")}
            if "action" in step: s["action"] = step["action"]
            steps.append(s)
        steps_str = json.dumps(steps, indent=2)
        
        context = f"Original Plan:\n{steps_str}"
        if history_str:
            context += f"\n\nExecution History (What actually happened):\n{history_str}"
            
        messages = [
            SystemMessage(content=STREAMLINE_PLAN_PROMPT),
            HumanMessage(content=context)
        ]
        
        try:
            response = await llm.ainvoke(messages)
            optimized_steps = parse_json_safe(response.content)
            if isinstance(optimized_steps, list):
                new_plan = []
                for s in optimized_steps:
                    if isinstance(s, str): new_plan.append({"step": s, "status": "pending"})
                    elif isinstance(s, dict):
                        item = {"step": s.get("description", ""), "status": "pending"}
                        if "action" in s: item["action"] = s["action"]
                        if "params" in s: item["params"] = s["params"]
                        new_plan.append(item)
                return new_plan
        except Exception as e:
            logger.error(f"Error streamlining plan: {e}")
        
        return current_plan # Return original if failure
