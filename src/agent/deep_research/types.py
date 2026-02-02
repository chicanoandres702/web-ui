from typing import List, Optional, TypedDict, Dict, Any
from pathlib import Path
from langchain_core.messages import BaseMessage
from langchain_core.tools import Tool

# Constants
REPORT_FILENAME = "report.md"
PLAN_FILENAME = "research_plan.md"
SEARCH_INFO_FILENAME = "search_info.json"

class ResearchTaskItem(TypedDict):
    task_description: str
    status: str  # "pending", "completed", "failed"
    queries: Optional[List[str]]
    result_summary: Optional[str]

class ResearchCategoryItem(TypedDict):
    category_name: str
    tasks: List[ResearchTaskItem]

class DeepResearchState(TypedDict):
    task_id: str
    topic: str
    research_plan: List[ResearchCategoryItem]
    search_results: List[Dict[str, Any]]
    llm: Any
    tools: List[Tool]
    output_dir: Path
    browser_config: Dict[str, Any]
    final_report: Optional[str]
    current_category_index: int
    current_task_index_in_category: int
    stop_requested: bool
    error_message: Optional[str]
    messages: List[BaseMessage]
    memory_file: Optional[str]
    google_docs_template_url: Optional[str]