"""
Utility scripts to be injected into the browser context.
These scripts clean the page of distractions before the agent acts.
"""
import logging
from src.utils.browser_scripts import JS_PURIFY_DOM

logger = logging.getLogger(__name__)

async def purify_page(page):
    """Removes ads and vignettes to prevent the agent from getting lost."""
    try:
        await page.evaluate(JS_PURIFY_DOM)
        return True
    except Exception as e:
        logger.error(f"Purification failed: {e}")
        return False