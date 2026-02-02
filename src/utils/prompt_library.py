"""
Library of popular and useful prompts for the Browser Use Agent.
"""
import os
import json
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

POPULAR_PROMPTS = [
    [
        "Academic Research (Google Scholar)",
        "Go to Google Scholar (scholar.google.com). Search for 'impact of artificial intelligence on healthcare'. For the top 3 results, extract the title, authors, and the APA citation (click the 'Cite' button). If a PDF is available, download it."
    ],
    [
        "YouTube Video Summary",
        "Go to YouTube. Search for 'Python asyncio tutorial'. Click on the first relevant video. Expand the description or look for a 'Show transcript' button. Extract the transcript text if available, or summarize the video description. Do not watch the video."
    ],
    [
        "News Aggregation (Hacker News)",
        "Navigate to 'https://news.ycombinator.com/'. Extract the titles and URLs of the top 10 stories. Save this list to a file named 'hackernews_top10.txt' in the downloads folder."
    ],
    [
        "Shopping Price Comparison",
        "Go to Amazon and search for 'Sony WH-1000XM5'. Find the current price. Then go to Best Buy and check the price for the same item. Tell me which one is cheaper and provide the links."
    ],
    [
        "Flight Search",
        "Go to Google Flights. Find a round-trip flight from New York (JFK) to London (LHR) for the dates June 1st to June 10th. List the cheapest option found including airline and price."
    ],
    [
        "Stock Market Check",
        "Go to Yahoo Finance. Search for 'NVDA' (Nvidia). Get the current stock price, the day's range, and the market cap."
    ],
    [
        "Interactive Quiz Solver",
        "Navigate to '[INSERT_QUIZ_URL]'. Start the quiz. For each question, analyze the options and select the most logical answer. Use 'scroll_slowly' to ensure you see all questions. If unsure, ask me for help."
    ],
    [
        "Google Docs Report Writing",
        "Navigate to 'https://docs.new'. Write a short report about 'The Future of Space Exploration'. Use 'Times New Roman', size 12. Create a title, introduction, and 3 bullet points. Ensure you handle any sign-in prompts if they appear."
    ]
]

CUSTOM_PROMPTS_DIR = "./tmp/custom_prompts"

def _get_safe_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip()

def get_all_prompts() -> List[Tuple[str, str]]:
    """Returns a list of (name, content) tuples combining default and custom prompts."""
    prompts = [(p[0], p[1]) for p in POPULAR_PROMPTS]
    
    if os.path.exists(CUSTOM_PROMPTS_DIR):
        for filename in os.listdir(CUSTOM_PROMPTS_DIR):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(CUSTOM_PROMPTS_DIR, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "name" in data and "content" in data:
                            prompts.append((f"[Custom] {data['name']}", data["content"]))
                except Exception as e:
                    logger.error(f"Error loading custom prompt {filename}: {e}")
    return prompts

def get_prompt_by_name(name: str) -> Optional[str]:
    """Retrieves content for a specific prompt name."""
    # Check defaults
    for p_name, p_content in POPULAR_PROMPTS:
        if p_name == name:
            return p_content
            
    # Check custom
    if name.startswith("[Custom] "):
        clean_name = name.replace("[Custom] ", "", 1)
        safe_name = _get_safe_filename(clean_name)
        filepath = os.path.join(CUSTOM_PROMPTS_DIR, f"{safe_name}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("content")
            except Exception:
                pass
    return None

def save_custom_prompt(name: str, content: str) -> bool:
    """Saves a custom prompt to disk."""
    try:
        os.makedirs(CUSTOM_PROMPTS_DIR, exist_ok=True)
        safe_name = _get_safe_filename(name)
        if not safe_name:
            return False
        filename = f"{safe_name}.json"
        filepath = os.path.join(CUSTOM_PROMPTS_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"name": name, "content": content}, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving custom prompt: {e}")
        return False

def delete_custom_prompt(name: str) -> bool:
    """Deletes a custom prompt from disk."""
    if not name.startswith("[Custom] "):
        return False
    
    clean_name = name.replace("[Custom] ", "", 1)
    safe_name = _get_safe_filename(clean_name)
    filename = f"{safe_name}.json"
    filepath = os.path.join(CUSTOM_PROMPTS_DIR, filename)
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
        else:
            logger.warning(f"Custom prompt file not found: {filepath}")
    except Exception as e:
        logger.error(f"Error deleting custom prompt: {e}")
    return False