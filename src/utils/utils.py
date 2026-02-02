import base64
import re
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests
import json
import gradio as gr
import uuid
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


def encode_image(img_path):
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data


def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """Get the latest recording and trace files"""
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files

    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")

    return latest_files


def read_file_safe(file_path: str) -> Optional[str]:
    """Safely read a file, returning None if it doesn't exist or on error."""
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def save_text_to_file(path: str, text: str, mode: str = "w"):
    """Safely save text to a file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)
    except Exception as e:
        logger.error(f"Error saving to file {path}: {e}")


async def save_text_to_file_async(path: str, text: str, mode: str = "w"):
    """Safely save text to a file asynchronously."""
    await asyncio.to_thread(save_text_to_file, path, text, mode)


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()


def save_to_knowledge_base_file(text: str, topic: str, memory_file_path: str) -> Optional[str]:
    """Appends text to a knowledge base file derived from the topic."""
    if not memory_file_path:
        return None
    base_dir = os.path.dirname(os.path.abspath(memory_file_path))
    safe_topic = sanitize_filename(topic) or "general"
    filename = f"kb_{safe_topic}.md"
    filepath = os.path.join(base_dir, filename)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f"\n## [{timestamp}] {topic}\n\n{text}\n"
    save_text_to_file(filepath, entry, mode="a")
    return filepath


def get_progress_bar_html(progress: int, label: str = "Progress") -> str:
    """Generates HTML for a progress bar."""
    return f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span style="margin-right: 10px; font-weight: bold;">{label}:</span>
        <div style="width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 1.5rem; overflow: hidden;">
            <div style="width: {progress}%; background-color: #3b82f6; height: 100%; border-radius: 9999px; transition: width 0.5s ease-in-out; text-align: center; color: white; line-height: 1.5rem; font-size: 0.875rem; font-weight: 600;">
                {progress}%
            </div>
        </div>
    </div>
    """


def parse_agent_thought(thought: str) -> Dict[str, str]:
    """Parses the agent's thought string into structured sections."""
    sections = {
        "Status": "",
        "Reasoning": "",
        "Challenge": "",
        "Analysis": "",
        "Next Steps": ""
    }

    if not thought:
        return sections

    # Normalize newlines
    thought = thought.replace('\r\n', '\n')

    # Regex for headers (flexible on bolding and case)
    header_pattern = re.compile(r'^(?:\*\*|#+\s*)?(Status|Reasoning|Challenge|Analysis|Next Steps)(?:\*\*|:)?\s*:', re.IGNORECASE | re.MULTILINE)

    parts = header_pattern.split(thought)

    if parts[0].strip():
        sections["Reasoning"] = parts[0].strip()

    for i in range(1, len(parts), 2):
        header = parts[i].title()
        content = parts[i+1].strip()
        for key in sections.keys():
            if key.lower() == header.lower():
                sections[key] = content
                break
    return sections
