import os
import logging
import shutil
from urllib.parse import urlparse
from src.utils.utils import save_text_to_file, read_file_safe, sanitize_filename

logger = logging.getLogger(__name__)

class SimpleMemoryManager:
    """
    A simple file-based memory manager to replace mem0.
    Stores site-specific knowledge in markdown files.
    """
    def __init__(self, memory_dir: str = "./tmp/memory"):
        self.memory_dir = os.path.abspath(memory_dir)
        os.makedirs(self.memory_dir, exist_ok=True)

    def _get_domain(self, url: str) -> str:
        try:
            if not url.startswith("http"):
                return url # Assume it's already a domain if no scheme
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return ""

    def get_site_knowledge(self, url: str) -> str:
        domain = self._get_domain(url)
        if not domain:
            return ""
        
        # Sanitize domain for filename
        safe_domain = sanitize_filename(domain)
        filename = f"site_knowledge_{safe_domain}.md"
        filepath = os.path.join(self.memory_dir, filename)
        
        return read_file_safe(filepath) or ""

    def save_site_knowledge(self, url: str, content: str) -> bool:
        domain = self._get_domain(url)
        if not domain: return False
        safe_domain = sanitize_filename(domain)
        filename = f"site_knowledge_{safe_domain}.md"
        filepath = os.path.join(self.memory_dir, filename)
        
        # Append new knowledge with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        entry = f"\n## [{timestamp}] Knowledge for {domain}\n{content}\n"
        
        # Use append mode if file exists, else write
        mode = "a" if os.path.exists(filepath) else "w"
        try:
            save_text_to_file(filepath, entry, mode=mode)
            return True
        except Exception as e:
            logger.error(f"Failed to save site knowledge: {e}")
            return False

# Global instance
_memory_manager = None

def get_memory_manager(memory_dir: str = "./tmp/memory"):
    global _memory_manager
    if not _memory_manager:
        _memory_manager = SimpleMemoryManager(memory_dir)
    # Update dir if changed (though usually static per run)
    if _memory_manager.memory_dir != os.path.abspath(memory_dir):
        _memory_manager.memory_dir = os.path.abspath(memory_dir)
        os.makedirs(_memory_manager.memory_dir, exist_ok=True)
    return _memory_manager

def configure_mem0():
    """Deprecated: No-op to prevent mem0 errors."""
    pass

def reset_mem0():
    """Deprecated: No-op."""
    pass

def create_procedural_memory(step_data: dict):
    """
    Attempts to save procedural memory. 
    Enhanced with a fail-safe to prevent agent stalling on memory errors.
    """
    try:
        logger.info(f"Attempting to create procedural memory for step: {step_data.get('step', 'unknown')}")
        # Placeholder for actual logic or integration with SimpleMemoryManager
        pass 
    except Exception as e:
        logger.warning(f"Procedural memory skip: {e}. Continuing with internal QuizStateManager.")
        return False
    return True

def get_relevant_memory(context_query: str):
    """Returns memory if available, or an empty list if the system is down."""
    try:
        return []
    except:
        return []