import base64
import re
import os
import time
from pathlib import Path
from typing import Dict, Optional, Callable, Any, List
import requests
import json
import gradio as gr
import uuid
import logging
import asyncio
from datetime import datetime
import sys
from src.utils.io_manager import IOManager


try:
    import json_repair
except ImportError:
    json_repair = None

logger = logging.getLogger(__name__)


def str_to_bool(val: Any) -> bool:
    """Converts a string representation of truth to True (1, y, yes, t, true, on) or False."""
    if isinstance(val, bool): return val
    if val is None: return False
    return str(val).lower() in ('y', 'yes', 't', 'true', 'on', '1')


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
    return IOManager.read_file_sync(file_path)


def save_text_to_file(path: str, text: str, mode: str = "w"):
    """Safely save text to a file."""
    IOManager.write_file_sync(path, text, mode=mode)


async def save_text_to_file_async(path: str, text: str, mode: str = "w"):
    """Safely save text to a file asynchronously."""
    await IOManager.write_file(path, text, mode=mode)



def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()

def resolve_file_path(filename: str) -> Optional[str]:

    """Helper to resolve file paths from common directories."""
    if not filename:
        return None
    
    # Check absolute path, downloads, uploads, and current working directory
    paths_to_check = [
        os.path.abspath(filename),
        os.path.join(os.path.abspath("./tmp/downloads"), os.path.basename(filename)),
        os.path.join(os.path.abspath("./tmp/uploads"), os.path.basename(filename)),
        os.path.join(os.getcwd(), os.path.basename(filename))
    ]
    
    return next((p for p in paths_to_check if os.path.exists(p)), None)

def extract_text_from_pdf(filepath: str) -> str:
    """Extracts text from a PDF file."""
    if not os.path.exists(filepath):
        return f"File not found: {filepath}"
        
    try:
        import pypdf
    except ImportError:
        return "Error: pypdf library not installed. Please install it via `pip install pypdf`."

    try:
        reader = pypdf.PdfReader(filepath)
        text = []
        for i, page in enumerate(reader.pages):
            extracted = page.extract_text()
            if extracted:
                text.append(f"--- Page {i+1} ---\n{extracted}")
        return "\n".join(text)
    except Exception as e:
        return f"Error reading PDF: {e}"


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
    IOManager.write_file_sync(filepath, entry, mode="a")
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



def calculate_progress_from_markdown(markdown_content: str) -> int:
    """Calculates progress percentage from markdown task lists."""
    if not markdown_content:
        return 0
    
    # Count task markers
    pending = markdown_content.count("- [ ]")
    completed = markdown_content.count("- [x]")
    failed = markdown_content.count("- [-]")
    
    total_tasks = pending + completed + failed
    processed_tasks = completed + failed
    
    if total_tasks == 0:
        return 0
        
    return min(100, int((processed_tasks / total_tasks) * 100))



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


async def retry_async(
    func: Callable[..., Any],
    *args,
    retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None,
    error_message: str = "Operation failed",
    **kwargs
) -> Any:
    """Retries an async function with exponential backoff."""
    last_exception = None
    current_delay = delay
    for attempt in range(retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(f"{error_message} (Attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(current_delay)
                current_delay *= backoff
    raise last_exception



def clean_json_string(content: str) -> str:
    """Cleans a string to extract JSON content, removing markdown code blocks and thinking traces."""
    # Remove <think> blocks (common in reasoning models)
    if "<think>" in content:
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    
    content = content.strip()
    
    # Attempt to find JSON start/end if extraneous text exists
    if content and not (content.startswith("{") or content.startswith("[")):
        match = re.search(r'(\{|\[)', content)
        if match:
            content = content[match.start():]
            
    if content and not (content.endswith("}") or content.endswith("]")):
        last_brace = content.rfind("}")
        last_bracket = content.rfind("]")
        end_index = max(last_brace, last_bracket)
        if end_index != -1:
            content = content[:end_index+1]
    
    # Robustness: If content looks like it was cut off (unterminated string), try to patch it
    # This is a heuristic for common LLM cutoff issues
    if content.count('"') % 2 != 0:
        if content.strip().endswith('"}'):
            pass # Might be okay
        else:
            content += '"}' # Attempt to close the JSON object
            
    return content


def extract_quoted_text(text: str) -> Optional[str]:
    """Extracts text inside single or double quotes."""
    if not text: return None
    match = re.search(r"['\"](.*?)['\"]", text)
    if match:
        return match.group(1)
    return ""


def parse_json_safe(content: str) -> Any:
    """
    Parses a JSON string, attempting to repair it if necessary.
    """
    cleaned = clean_json_string(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        if json_repair:
            try:
                return json_repair.loads(cleaned)
            except Exception:
                pass
        raise


async def run_tasks_in_parallel(
    task_factories: List[Callable[[], Any]],
    max_concurrent: int = 5,
    return_exceptions: bool = True
) -> List[Any]:
    """Runs a list of async task factories (callables returning coroutines) with a concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrent)
    async def worker(factory):
        async with semaphore:
            return await factory()
    return await asyncio.gather(*(worker(f) for f in task_factories), return_exceptions=return_exceptions)



def ensure_default_extraction_models():
    """Creates default extraction model JSONs if they don't exist."""
    models_dir = "./tmp/extraction_models"
    os.makedirs(models_dir, exist_ok=True)
    
    default_models = [
        {
            "name": "ProductDetails",
            "fields": [
                {"name": "product_name", "type": "str", "description": "The name of the product"},
                {"name": "price", "type": "float", "description": "The price of the product"},
                {"name": "currency", "type": "str", "description": "The currency of the price (e.g., USD, EUR)"},
                {"name": "availability", "type": "str", "description": "Availability status (e.g., In Stock, Out of Stock)"},
                {"name": "rating", "type": "float", "description": "Product rating (0-5)"},
                {"name": "review_count", "type": "int", "description": "Number of reviews"}
            ]
        },
        {
            "name": "JobPosting",
            "fields": [
                {"name": "job_title", "type": "str", "description": "Title of the job position"},
                {"name": "company", "type": "str", "description": "Name of the hiring company"},
                {"name": "location", "type": "str", "description": "Job location"},
                {"name": "salary_range", "type": "str", "description": "Salary range if available"},
                {"name": "description", "type": "str", "description": "Brief summary of the job description"},
                {"name": "requirements", "type": "List[str]", "description": "List of key requirements or skills"}
            ]
        },
        {
            "name": "NewsArticle",
            "fields": [
                {"name": "headline", "type": "str", "description": "The main headline of the article"},
                {"name": "author", "type": "str", "description": "Name of the author"},
                {"name": "publication_date", "type": "str", "description": "Date of publication"},
                {"name": "summary", "type": "str", "description": "A concise summary of the article content"},
                {"name": "topics", "type": "List[str]", "description": "List of main topics or tags"}
            ]
        },
        {
            "name": "SearchResult",
            "fields": [
                {"name": "title", "type": "str", "description": "Title of the search result"},
                {"name": "url", "type": "str", "description": "URL of the result"},
                {"name": "snippet", "type": "str", "description": "Snippet or description text"}
            ]
        },
        {
            "name": "Recipe",
            "fields": [
                {"name": "name", "type": "str", "description": "Name of the recipe"},
                {"name": "ingredients", "type": "List[str]", "description": "List of ingredients"},
                {"name": "prep_time", "type": "str", "description": "Preparation time"},
                {"name": "cook_time", "type": "str", "description": "Cooking time"},
                {"name": "servings", "type": "int", "description": "Number of servings"}
            ]
        },
        {
            "name": "Event",
            "fields": [
                {"name": "title", "type": "str", "description": "Title of the event"},
                {"name": "date", "type": "str", "description": "Date and time of the event"},
                {"name": "location", "type": "str", "description": "Location or venue"},
                {"name": "description", "type": "str", "description": "Description of the event"},
                {"name": "price", "type": "str", "description": "Ticket price or cost"}
            ]
        }
    ]

    for model in default_models:
        filename = f"{model['name']}.json"
        filepath = os.path.join(models_dir, filename)
        if not os.path.exists(filepath):
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(model, f, indent=2)
            except Exception as e:
                logger.error(f"Failed to create default extraction model {filename}: {e}")


def suppress_asyncio_cleanup_errors():
    """
    Suppresses 'I/O operation on closed pipe' errors on Windows during asyncio cleanup.
    This is a known issue with ProactorEventLoop on Windows when processes are terminated abruptly.
    """
    if sys.platform == 'win32':
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport
            
            # Monkey-patch __del__ to silence the ValueError
            def silence_del(self):
                if hasattr(self, '_sock') and self._sock is not None:
                    try: self._sock.close()
                    except: pass
            _ProactorBasePipeTransport.__del__ = silence_del
        except (ImportError, AttributeError):
            pass
