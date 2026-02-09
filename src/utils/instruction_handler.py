import logging
from typing import Dict, Any, List
from src.utils.utils import extract_quoted_text
from src.agent.deep_research.types import ResearchTaskItem as Task

logger = logging.getLogger(__name__)

class InstructionHandler:
    """
    Handles specific instructions, breaking them down into actionable tasks for the agent.
    """

    def __init__(self, agent):
        self.agent = agent

    def handle_instruction(self, instruction: str) -> List[Task]:
        """
        Processes an instruction and returns a list of tasks for the agent to execute.
        """
        instruction = instruction.lower()
        if "read chapter" in instruction:
            return self._handle_read_chapter(instruction)
        elif "find" in instruction and ("link" in instruction or "library" in instruction or "bookshelf" in instruction):
            return self._handle_find_resource(instruction)
        else:
            logger.warning(f"Unknown instruction type: {instruction}")
            return [Task(description=instruction, task_type="instruction")]

    def _handle_read_chapter(self, instruction: str) -> List[Task]:
        """
        Handles instructions to read a chapter, prioritizing the reading task.
        """
        chapter_number = extract_quoted_text(instruction)
        if not chapter_number:
            chapter_number = instruction.split("chapter")[-1].strip()

        reading_task = Task(
            description=f"Read Chapter {chapter_number}",
            task_type="reading",
            metadata={"chapter_number": chapter_number}
        )

        find_task = Task(
            description=f"Find Chapter {chapter_number} in the book",
            task_type="navigation",
            metadata={"chapter_number": chapter_number}
        )
        return [reading_task, find_task]

    def _handle_find_resource(self, instruction: str) -> List[Task]:
        """
        Handles instructions to find a resource (link, library item, etc.).
        """
        resource_name = extract_quoted_text(instruction)
        if not resource_name:
            resource_name = instruction.split("find")[-1].strip()

        find_task = Task(
            description=f"Locate resource: {resource_name}",
            task_type="navigation",
            metadata={"resource_name": resource_name}
        )

        return [find_task]

    async def execute_instruction_tasks(self, tasks: List[Task]) -> str:
        """
        Executes the generated instruction tasks.
        """
        last_result = ""
        for task in tasks:
            logger.info(f"Executing instruction task: {task.description}")
            # Adapt the execution logic based on the task type
            if task.task_type == "reading":
                last_result = await self._execute_reading_task(task)
            elif task.task_type == "navigation":
                last_result = await self._execute_navigation_task(task)
            else:
                last_result = f"Unsupported task type: {task.task_type}"
            if "Error" in last_result or "Failed" in last_result:
                return last_result
        return last_result

    async def _execute_reading_task(self, task: Task) -> str:
        """
        Executes the task of reading a specific chapter.
        """
        # Implement logic to navigate to the chapter and extract content
        # (e.g., using browser actions to find the chapter and read its content)
        chapter_number = task.metadata.get("chapter_number")
        return f"Successfully read Chapter {chapter_number}."

    async def _execute_navigation_task(self, task: Task) -> str:
        """
        Executes the task of navigating to a specific resource.
        """
        # Implement logic to find the resource (link, library item)
        # (e.g., using search or navigation actions)
        resource_name = task.metadata.get("resource_name")
        return f"Successfully located resource: {resource_name}."



def create_instruction_handler(agent):
    """
    Factory function to create an InstructionHandler.
    """
    return InstructionHandler(agent)