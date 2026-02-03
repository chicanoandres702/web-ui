import logging
from urllib.parse import urlparse
from typing import List, Optional

logger = logging.getLogger(__name__)

class NavigationController:
    """
    Evaluates navigation requests. Instead of hard-blocking, it flags 
    suspicious domains for the agent to review.
    """
    def __init__(self, allowed_domains: Optional[List[str]] = None):
        self.allowed_domains = allowed_domains or ["google.com", "wikipedia.org"]
        # List of domains that usually trigger a 'distraction check'
        self.suspicious_domains = ["facebook.com", "twitter.com", "instagram.com", "reddit.com", "youtube.com"]

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

    async def navigate(self, context, url: str):
        """Standard navigation wrapper."""
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            return page
        except Exception as e:
            logger.error(f"Navigation failed to {url}: {e}")
            raise
