import os
import logging
from typing import List
from src.utils.utils import read_file_safe

logger = logging.getLogger(__name__)

def list_kb_files(dir_path: str) -> List[str]:
    """Lists all markdown and text files in the directory recursively."""
    if not dir_path:
        return []
    
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception:
            return []
            
    try:
        files = []
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith(".md") or filename.endswith(".txt"):
                    rel_path = os.path.relpath(os.path.join(root, filename), dir_path)
                    rel_path = rel_path.replace("\\", "/")
                    files.append(rel_path)
        files.sort()
        return files
    except Exception as e:
        logger.error(f"Error listing KB files: {e}")
        return []

def search_kb_files(dir_path: str, query: str) -> List[str]:
    """Searches for a query string within files in the directory."""
    if not dir_path or not os.path.exists(dir_path):
        return []
    
    if not query:
        return list_kb_files(dir_path)
        
    matches = []
    try:
        for root, _, filenames in os.walk(dir_path):
            for filename in filenames:
                if not (filename.endswith(".md") or filename.endswith(".txt")):
                    continue
                    
                path = os.path.join(root, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        if query.lower() in file.read().lower():
                            rel_path = os.path.relpath(path, dir_path)
                            rel_path = rel_path.replace("\\", "/")
                            matches.append(rel_path)
                except Exception: pass
        return sorted(matches)
    except Exception as e:
        logger.error(f"Search error: {e}")
        return []

def load_kb_content(dir_path: str, filename: str) -> str:
    """Loads content of a specific file."""
    if not dir_path or not filename:
        return ""
    filepath = os.path.join(dir_path, filename)
    return read_file_safe(filepath) or ""
