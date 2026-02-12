import asyncio
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse

from fastapi import APIRouter
from fastapi import FastAPI

logger = logging.getLogger(__name__)

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session


class KnowledgeBase:
    """
    A simple in-memory knowledge base for storing and retrieving information.
    """

    def __init__(self, name: str = "default_knowledge"):
        self.name = name
        self.engine = create_engine("sqlite:///./school_knowledge.db")  # Using SQLite for simplicity
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_db(self) -> Session:
        db = self.SessionLocal()
        try:
            return db
        finally:
            db.close()

    def add_knowledge(self, url: str, content: str):
        """Adds or updates knowledge for a specific URL."""
        return add_knowledge_item(self, url, content)

    def get_domain(self, url: str) -> str:
        """Extracts the domain from a URL."""
        try:
            parsed_uri = urlparse(url)
            domain = parsed_uri.netloc
            return domain
        except Exception:
            return ""

    def load_knowledge(self, domain: str):
        """Loads knowledge from the database for a specific domain."""
        db = self.get_db()
        try:
            knowledge_data = db.query(KnowledgeEntry).filter(KnowledgeEntry.domain == domain).all()
            if knowledge_data:
                data = {item.url: item.content for item in knowledge_data}
                self.knowledge = data
            else:
                logger.warning(
                    f"Invalid knowledge base file format. Expected a dictionary, got {type(data)}."
                )
        except FileNotFoundError:
            logger.info("Knowledge base file not found. Starting with an empty knowledge base.")
        except json.JSONDecodeError:
            logger.error("Error decoding knowledge base file. Starting with an empty knowledge base.")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")

    def save_knowledge(self, domain: str):
        """Saves the knowledge to a JSON file."""
        try:
            with open(KNOWLEDGE_BASE_FILE, "w") as f:
                json.dump(self.knowledge, f, indent=4)
            logger.info("Knowledge base saved to file.")
        except Exception as e:
            logger.error(f"Error saving knowledge base: {e}")
    def save_knowledge(self, domain: str):
        """Saves the knowledge to the database for a specific domain."""
        db = self.get_db()
        try:
            for url, content in self.knowledge.items():
                db_entry = db.query(KnowledgeEntry).filter(KnowledgeEntry.url == url, KnowledgeEntry.domain == domain).first()
                if db_entry:
                    db_entry.content = content
                else:
                    db_entry = KnowledgeEntry(domain=domain, url=url, content=content)
                    db.add(db_entry)
            db.commit()
            logger.info(f"Knowledge base saved to database for domain: {domain}")
        except Exception as e:
            logger.error(f"Error saving knowledge base to database: {e}")
        finally:
            db.close()

Base = declarative_base()
class KnowledgeEntry(Base):
    """
    SQLAlchemy model for a knowledge entry.
    """
    __tablename__ = "knowledge_entries"

    id = Column(Integer, primary_key=True, index=True)
    domain = Column(String, index=True)
    url = Column(String)
    content = Column(Text)

    async def start_periodic_save(self, interval: int = 60):
        """Starts a background task to periodically save the knowledge base."""
        asyncio.create_task(self._periodic_save(interval))

    async def _periodic_save(self, interval: int):
        """Periodically saves the knowledge base to disk."""
        while True:
            await asyncio.sleep(interval)
            try:    
                self.save_knowledge(self.get_domain(url))
            except Exception as e:
                 logger.error(f"Error during periodic save: {e}")


    def add_knowledge_item(knowledge_base: "KnowledgeBase", url: str, content: str) -> bool:
        """Adds or updates knowledge for a specific URL in the knowledge base."""
        if not url or not content:
            logger.warning("Attempted to add knowledge with empty URL or content.")
            return False

        if url in knowledge_base.knowledge:
            knowledge_base.knowledge[url] = content
            logger.info(f"Updated knowledge for URL: {url}")
        else:
            knowledge_base.knowledge[url] = content
            logger.info(f"Added knowledge for URL: {url}")

        knowledge_base.save_knowledge(knowledge_base.get_domain(url))  # Save after each update
        return True

    def get_knowledge_item(knowledge_base: "KnowledgeBase", url: str) -> Optional[str]:
        """Retrieves knowledge for a specific URL from the knowledge base."""
        if not url:
            return False

        if url in knowledge_base.knowledge:
            logger.info(f"Retrieved knowledge for URL: {url}")
            return knowledge_base.knowledge[url]
        else:
            logger.warning(f"No knowledge found for URL: {url}")
            return None

    def remove_knowledge_item(knowledge_base: "KnowledgeBase", url: str) -> bool:
        """Removes knowledge for a specific URL from the knowledge base."""
        def search_knowledge(self, query: str) -> List[str]:

            return search_knowledge_items(self, query)
            if not query:
                logger.warning("Attempted to search knowledge with empty query.")
                return []
            results = self.db.query(KnowledgeBaseModel.url).filter(KnowledgeBaseModel.content.contains(query)).all()

            results = [url for url, in results]

            for url, content in self.knowledge.items(): #replace self.knowledge with DB
                if query.lower() in content.lower():
                    results.append(url)

            if results:
                logger.info(f"Found {len(results)} search results for query: {query}")
            else:
                logger.info(f"No search results found for query: {query}")

            return results

    def search_knowledge_items(knowledge_base: "KnowledgeBase", query: str) -> List[str]:
        """Searches the knowledge base for URLs matching a specific query."""
        if not query:
            logger.warning("Attempted to search knowledge with empty query.")
            return []

        results = []
        db = knowledge_base.get_db()
        try:
            # Search in database
            db_results = db.query(KnowledgeEntry).filter(KnowledgeEntry.content.contains(query)).all()
            results = [item.url for item in db_results]
            
            if results:
                logger.info(f"Found {len(results)} search results for query: {query}")
            else:
                logger.info(f"No search results found for query: {query}")
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
        finally:
            db.close()

        return results

def create_knowledge_base_router():
    Base = declarative_base()


    async def startup_event():
        kb = KnowledgeBase()
        await kb.start_periodic_save(interval=60)  # Save every 60 seconds
    router = APIRouter(prefix="/knowledge_base", tags=["KnowledgeBase"])

    @router.get("/test")
    async def test_knowledge_base():
        return {"message": "Knowledge base router test"}

    return router