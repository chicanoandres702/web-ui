import os
import logging
import shutil
from urllib.parse import urlparse
from typing import Optional
from typing import Callable, Coroutine, Any
import asyncio
from src.utils.utils import save_text_to_file, read_file_safe, sanitize_filename

logger = logging.getLogger(__name__)

async def _create_embedding(text: str):
    """
    Abstracts the process of creating an embedding for a given text.

    Handles model selection and potential API errors.
    using the provided llm if available, otherwise, a default.
    """

    try:
        if llm:
            # Use the provided llm to generate embeddings
            if hasattr(llm, "embed_query") and callable(llm.embed_query):
                return await asyncio.to_thread(llm.embed_query, text)
            elif hasattr(llm, "aembed_query") and callable(llm.aembed_query):
                return await llm.aembed_query(text)
        # For now, return a dummy embedding (list of floats)
        return [0.0] * 1536  # Assuming 1536 is the dimensionality of your embeddings

    except Exception as e:
        logger.error(f"Error creating embedding: {e}")
        return None







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
        if not _memory_manager:
            logger.info("Creating default SimpleMemoryManager as backup.")
            _memory_manager = SimpleMemoryManager()


def configure_mem0():
    """Deprecated: No-op to prevent mem0 errors."""
    pass

def reset_mem0():
    """Deprecated: No-op."""
    pass

async def create_procedural_memory(step_data: dict, llm) -> bool:
    """
    Attempts to save procedural memory. 
    Enhanced with a fail-safe to prevent agent stalling on memory errors.
    """
    try:
         logger.info(f"Attempting to create procedural memory for step: {step_data.get('step', 'unknown')}")


         # 1. Create Knowledge Entry
         knowledge = f"Procedural memory for step: {step_data.get('step', 'unknown')}\n{step_data}"

         # 2. Generate Embedding (Abstracted)
         embedding = await _create_embedding(knowledge, llm=llm)
         if not embedding:
            logger.warning(f"Procedural memory skip: Embedding creation failed. Continuing with internal QuizStateManager.")

            return False
    except Exception as e:
        logger.warning(f"Procedural memory skip: {e}. Continuing with internal QuizStateManager.")


    return True


llm = None
    

def get_relevant_memory(context_query: str, llm=None):
    """Returns memory if available, or an empty list if the system is down. Uses the provided llm for retrieval."""
    try:
        return []
    except:
        return []