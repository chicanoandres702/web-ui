import logging
from urllib.parse import urlparse
from typing import List, Optional
from src.utils.config import DISTRACTION_DOMAINS, EXTRANEOUS_URL_PATTERNS

logger = logging.getLogger(__name__)

class NavigationController:
    """
    Evaluates navigation requests. Instead of hard-blocking, it flags 
    suspicious domains for the agent to review.
    """
    def __init__(self, allowed_domains: Optional[List[str]] = None):
        self.allowed_domains = allowed_domains or ["google.com", "wikipedia.org"]
        # List of domains that usually trigger a 'distraction check'
        self.suspicious_domains = DISTRACTION_DOMAINS

    def evaluate_url(self, url: str) -> str:
        """
        Returns 'SAFE', 'SUSPICIOUS', or 'EXTERNAL'.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if not domain:
                return "EXTERNAL" # Treat relative or invalid URLs as external/unknown
            
            # Check allowed first
            if any(allowed in domain for allowed in self.allowed_domains):
                return "SAFE"
                
            # Check suspicious
            if any(suspicious in domain for suspicious in self.suspicious_domains):
                return "SUSPICIOUS"
                
            return "EXTERNAL"
        except Exception as e:
            logger.error(f"Error evaluating URL {url}: {e}")
            return "EXTERNAL"

    async def manage_tabs(self, context):
        """Closes extraneous tabs to keep the agent focused."""
        try:
            if not context or not hasattr(context, 'context'):
                return
                
            pages = context.context.pages
            if len(pages) > 1:
                active_page = pages[-1]
                active_url = active_page.url.lower()
                
                for i in range(len(pages) - 2, -1, -1):
                    page = pages[i]
                    url = page.url.lower()
                    
                    if url == active_url:
                        logger.info(f"Closing duplicate tab: {url}")
                        await page.close()
                    elif any(d in url for d in self.suspicious_domains) or any(p in url for p in EXTRANEOUS_URL_PATTERNS):
                        logger.info(f"Closing extraneous tab: {url}")
                        await page.close()
                    elif len(context.context.pages) > 5:
                        logger.info(f"Closing background tab to save resources: {url}")
                        await page.close()
        except Exception as e:
            logger.warning(f"Error managing tabs: {e}")

    async def navigate(self, context, url: str):
        """Standard navigation wrapper."""
        try:
            if not hasattr(context, 'context'):
                raise AttributeError("'CustomBrowserContext' object has no attribute 'context'")
            
            page = context.context.pages[0] if context.context.pages else await context.context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            return page
        except Exception as e:
            logger.error(f"Navigation failed to {url}: {e}")
            raise