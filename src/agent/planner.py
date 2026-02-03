import json
import logging
from typing import List, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from src.utils.prompts import DEFAULT_PLANNER_PROMPT
from src.utils.utils import clean_json_string

logger = logging.getLogger(__name__)

async def generate_hierarchical_plan(llm: BaseChatModel, task: str, system_prompt: Optional[str] = None) -> List[str]:
    """
    Generates a hierarchical plan (list of subtasks) for the given task using the provided LLM.
    """
    prompt_content = system_prompt if system_prompt and system_prompt.strip() else DEFAULT_PLANNER_PROMPT
    
    messages = [
        SystemMessage(content=prompt_content),
        HumanMessage(content=f"User Task: {task}")
    ]
    
    try:
        response = await llm.ainvoke(messages)
        content = clean_json_string(response.content)
        
        if not content:
            logger.warning("Planner returned empty content.")
            return []
        
        plan = json.loads(content)
        if isinstance(plan, list) and all(isinstance(item, str) for item in plan):
            return plan
        else:
            logger.warning(f"Planner output was not a list of strings: {content}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error in planner: {e}. Content: {content[:100]}...")
        return []
    except Exception as e:
        logger.error(f"Error generating plan: {e}")
        return []