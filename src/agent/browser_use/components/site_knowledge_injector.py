import asyncio
import logging
from typing import TYPE_CHECKING

from src.agent.browser_use.components.knowledge_base import KnowledgeBase

if TYPE_CHECKING:
    from src.agent.browser_use.browser_use_agent import BrowserUseAgent


logger = logging.getLogger(__name__)

class SiteKnowledgeInjector:
    """
    Manages the injection of site-specific knowledge into the agent's memory,
    enhancing its understanding of the current website.
    """

    def __init__(self, agent: "BrowserUseAgent"):
        self.agent = agent
        self.knowledge_base = KnowledgeBase(name="SiteKnowledge")

    async def inject_site_knowledge(self) -> None:
        """
        Retrieves and<ctrl63> injects relevant site knowledge into the agent's memory.
        """
        if not getattr(self.agent, 'use_custom_memory', False) or not self.agent.memory_manager:
            return

        try:
            if not self.agent.browser_context:
                return

            page = await self.agent.browser_context.get_current_page()
            url = page.url
            domain = url.split('/')[2] if '/' in url else url

            # Only inject if the domain has changed to avoid redundant memory operations
            if domain != getattr(self.agent.step_handler, 'last_domain', None):
                try:
                    knowledge = await self.agent.memory_manager.get_knowledge_for_domain(domain)
                    if knowledge:
                        if self.knowledge_base.store_knowledge(domain, knowledge):
                            self.agent.heuristics.inject_message(f"SYSTEM: Site-specific knowledge for {domain}: {knowledge}")
                            logger.info(f"Successfully injected site knowledge for {domain}")
                        else:
                            logger.warning(f"Failed to store knowledge in knowledge base for domain: {domain}")
                    else:
                        logger.info(f"No site knowledge found for {domain}")
                    self.agent.step_handler.last_domain = domain
                except Exception as e:
                    logger.error(f"Error while retrieving/injecting knowledge for domain {domain}: {e}")

        except Exception as e:
            logger.error(f"Failed to inject site knowledge: {e}")