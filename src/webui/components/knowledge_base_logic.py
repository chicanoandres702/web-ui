import os
import logging
from typing import List, Tuple, Any
from src.webui.components.shared import read_text_file, save_text_file, rename_file, save_uploaded_file
from datetime import datetime
from src.utils.knowledge_base import list_kb_files, search_kb_files, load_kb_content

logger = logging.getLogger(__name__)

# Re-exporting functions from src.utils.knowledge_base for backward compatibility
list_kb_files = list_kb_files
search_kb_files = search_kb_files
load_kb_content = load_kb_content

def save_kb_content(dir_path: str, filename: str, content: str) -> Tuple[bool, str]:
    """Saves content to a specific file."""
    if not dir_path or not filename:
        return False, "No file selected."
    filepath = os.path.join(dir_path, filename)
    if save_text_file(filepath, content):
        return True, f"✅ Saved {filename} at {datetime.now().strftime('%H:%M:%S')}"
    return False, "Error saving file."

def create_kb_file(dir_path: str, filename: str) -> Tuple[bool, str, str]:
    """Creates a new file."""
    if not dir_path or not filename:
        return False, "Invalid directory or filename.", ""
        
    if not filename.endswith(".md") and not filename.endswith(".txt"):
        filename += ".md"
        
    filepath = os.path.join(dir_path, filename)
    if os.path.exists(filepath):
        return False, f"File '{filename}' already exists.", ""
        
    if save_text_file(filepath, "# " + filename):
        return True, f"✅ Created '{filename}'.", filename
    logger.error(f"Error creating file at path: {filepath}")
    return False, "Error creating file.", ""

def delete_kb_file(dir_path: str, filename: str) -> Tuple[bool, str]:
    """Deletes a file."""
    if not dir_path or not filename:
        return False, "No file selected."
    filepath = os.path.join(dir_path, filename)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True, f"🗑️ Deleted '{filename}'."
        return False, "File not found."
    except Exception as e:
        return False, f"Error deleting file: {e}"

def rename_kb_file(dir_path: str, filename: str, new_name: str) -> Tuple[bool, str, str]:
    """Renames a file."""
    if not dir_path or not filename or not new_name:
        return False, "Invalid parameters.", ""
    old_path = os.path.join(dir_path, filename)
    new_path = os.path.join(dir_path, new_name)
    if rename_file(old_path, new_path):
        return True, f"✅ Renamed '{filename}' to '{new_name}'.", new_name
    return False, "Error renaming file.", ""

def import_kb_files(dir_path: str, files: List[Any]) -> Tuple[bool, str]:
    """Imports uploaded files."""
    if not dir_path or not files:
        return False, "No files to import."
    count = 0
    for file_obj in files:
        try:
            temp_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
            original_name = os.path.basename(temp_path)
            dest_path = os.path.join(dir_path, original_name)
            if save_uploaded_file(temp_path, dest_path):
                count += 1
        except Exception as e:
            logger.error(f"Import error: {e}")
    return True, f"✅ Imported {count} files."