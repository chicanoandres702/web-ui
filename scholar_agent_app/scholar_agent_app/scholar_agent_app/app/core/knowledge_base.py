import json
import os
from app.config import get_settings
settings = get_settings()

class KnowledgeManager:
    def __init__(self):
        self.base_dir = settings.KNOWLEDGE_BASE_DIR
    def _get_path(self, class_name: str) -> str:
        safe_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '-', '_')).strip()
        return os.path.join(self.base_dir, f"{safe_name}.json")
    def load(self, class_name: str) -> dict:
        path = self._get_path(class_name)
        if os.path.exists(path):
            with open(path, 'r') as f: return json.load(f)
        return {}
    def save(self, class_name: str, data: dict):
        current = self.load(class_name)
        current.update(data)
        with open(self._get_path(class_name), 'w') as f: json.dump(current, f, indent=2)

kb_manager = KnowledgeManager()
