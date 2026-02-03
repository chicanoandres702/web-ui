"""
Persistent state management for the agent's progress.
"""
import json
import logging
import os
import re
from typing import Dict, Any, Optional
from src.utils.utils import save_text_to_file, read_file_safe

logger = logging.getLogger(__name__)

class QuizStateManager:
    """
    Persistent state management for the agent's progress in quizzes.
    """
    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = persistence_path
        self.state = {
            "current_question": 0,
            "total_questions": 0,
            "completed_steps": [],
            "last_known_anchor": None,
            "is_interrupted": False
        }
        if self.persistence_path:
            self._load_state()

    def update_progress(self, question_num: int, total: int = 10, anchor: str = None):
        self.state["current_question"] = question_num
        self.state["total_questions"] = total
        if anchor:
            self.state["last_known_anchor"] = anchor
        self.state["is_interrupted"] = False
        self.save_to_log()

    def mark_interrupted(self):
        self.state["is_interrupted"] = True
        self.save_to_log()

    def get_recovery_context(self) -> str:
        if self.state["current_question"] > 0:
            anchor_msg = f" Look for anchor '{self.state['last_known_anchor']}'." if self.state['last_known_anchor'] else ""
            return f"Resume from Question {self.state['current_question']} of {self.state['total_questions']}.{anchor_msg}"
        return "Start quiz from the beginning."

    def sync_from_page(self, page_text: str):
        """Parses page content to update state."""
        # Look for "Question X of Y" pattern
        match = re.search(r"Question\s+(\d+)\s*(?:of|/)\s*(\d+)", page_text, re.IGNORECASE)
        if match:
            current, total = map(int, match.groups())
            self.update_progress(current, total)

    def get_status_summary(self) -> str:
        return f"Q: {self.state['current_question']}/{self.state['total_questions']} | Interrupted: {self.state['is_interrupted']}"

    def save_to_log(self):
        """Saves current state to disk if a path is configured."""
        if self.persistence_path:
            try:
                save_text_to_file(self.persistence_path, json.dumps(self.state, indent=2))
            except Exception as e:
                logger.error(f"Failed to save quiz state: {e}")

    def _load_state(self):
        """Loads state from disk."""
        if self.persistence_path and os.path.exists(self.persistence_path):
            content = read_file_safe(self.persistence_path)
            if content:
                try:
                    loaded = json.loads(content)
                    self.state.update(loaded)
                    logger.info(f"Loaded quiz state: Question {self.state['current_question']}")
                except json.JSONDecodeError:
                    logger.warning("Failed to decode quiz state file.")