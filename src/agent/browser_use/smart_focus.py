import logging
import asyncio
from src.utils.browser_scripts import JS_SMART_SCROLL

logger = logging.getLogger(__name__)

async def focus_on_content(page, semantic_query: str) -> bool:
    """Triggers the semantic scroll and visual anchoring with internal retries."""
    if not semantic_query:
        return False
        
    # Attempt to find the content immediately, then retry after a short wait if lazy-loading
    for attempt in range(3):
        try:
            success = await page.evaluate(JS_SMART_SCROLL, semantic_query)
            if success:
                logger.info(f"Focused on content matching: '{semantic_query}'")
                return True
        except Exception:
            pass
        await page.wait_for_timeout(1500) # Wait for potential dynamic content
    
    logger.info(f"Could not find content matching: '{semantic_query}'")
    return False

async def wait_for_text(page, text: str, timeout: int = 5000):
    """Wait specifically for a piece of text to appear in the DOM."""
    try:
        await page.wait_for_function(
            f"() => document.body.innerText.toLowerCase().includes('{text.lower()}')",
            timeout=timeout
        )
        return True
    except Exception:
        return False