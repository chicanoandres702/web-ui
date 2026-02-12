import os
import re

def save_text_to_file(filepath: str, text: str, mode: str = "w") -> None:
    """Saves the given text to a file."""
    try:
        with open(filepath, mode, encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        raise Exception(f"Failed to write to file: {e}")

def read_file_safe(filepath: str) -> str:
    """Safely reads the content of a file, handling potential errors."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^\w.-]", "_", filename)