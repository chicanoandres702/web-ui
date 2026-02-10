import os
import logging
from typing import List
from src.utils.utils import read_file_safe, save_text_to_file, sanitize_filename
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    """
    Manages multiple knowledge bases, each associated with a specific domain.
    Provides methods to store and retrieve knowledge for each website.
    """

    def __init__(self, base_dir: str = "./tmp/memory"):
        """
        Initializes the KnowledgeBaseManager.

        Args:
            base_dir (str): The base directory where knowledge base files will be stored.
        """
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"KnowledgeBaseManager initialized with base directory: {self.base_dir}")

    def _get_domain(self, url: str) -> str:
        """
        Extracts the domain name from a URL.

        Args:
            url (str): The URL to extract the domain from.

        Returns:
            str: The domain name, or an empty string if extraction fails.
        """
        try:
            if not url.startswith("http"):
                return url  # Assume it's already a domain if no scheme
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return ""

    def get_site_knowledge(self, url: str) -> str:
        """
        Retrieves knowledge stored for a specific website.

        Args:
            url (str): The URL of the website.

        Returns:
            str: The stored knowledge, or an empty string if no knowledge is found.
        """
        domain = self._get_domain(url)
        if not domain:
            return ""

        safe_domain = sanitize_filename(domain)
        filename = f"site_knowledge_{safe_domain}.md"
        filepath = os.path.join(self.base_dir, filename)

        content = read_file_safe(filepath)
        if content:
            logger.info(f"Retrieved knowledge for domain '{domain}' from '{filepath}'.")
            return content
        logger.info(f"No knowledge found for domain '{domain}'.")
        return ""