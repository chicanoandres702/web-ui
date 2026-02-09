import logging
from typing import Dict, Optional, Any
import json

logger = logging.getLogger(__name__)

class KnowledgeBase:
    """
    Manages a knowledge base, providing methods to store, retrieve, and process knowledge.
    """

    def __init__(self, name: str, storage: Optional[Dict[str, Any]] = None):
        """
        Initializes the KnowledgeBase.

        Args:
            name (str): The name of the knowledge base.
            storage (Optional[Dict[str, Any]]): Initial storage for the knowledge base. Defaults to an empty dictionary.
        """
        self.name = name
        self.storage = storage if storage is not None else {}
        logger.info(f"KnowledgeBase '{self.name}' initialized.")

    def store_knowledge(self, key: str, knowledge: Any) -> bool:
        """
        Stores knowledge in the knowledge base.

        Args:
            key (str): The key to store the knowledge under.
            knowledge (Any): The knowledge to store.

        Returns:
            bool: True if the knowledge was successfully stored, False otherwise.
        """
        try:
            if not isinstance(key, str):
                raise ValueError("Key must be a string.")
            self.storage[key] = knowledge
            logger.info(f"Knowledge stored successfully under key '{key}'.")
            return True
        except ValueError as e:
            logger.error(f"Failed to store knowledge under key '{key}': {e}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while storing knowledge under key '{key}': {e}")
            return False

    def retrieve_knowledge(self, key: str) -> Any:
        """
        Retrieves knowledge from the knowledge base.

        Args:
            key (str): The key to retrieve the knowledge from.

        Returns:
            Any: The knowledge retrieved, or None if the key is not found.
        """
        try:
            return self.storage.get(key)
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge for key '{key}': {e}")

    def import_knowledge(self, filepath: str) -> bool:
        """
        Imports knowledge from a JSON file into the knowledge base.

        Args:
            filepath (str): The path to the file where the knowledge base is stored.

        Returns:
            bool: True if the import was successful, False otherwise.
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    self.storage.update(data)
                    logger.info(f"Knowledge base successfully imported from '{filepath}'.")
                    return True
                else:
                    raise ValueError("The file does not contain a valid JSON dictionary.")
        except Exception as e:
            logger.error(f"Failed to import knowledge base from '{filepath}': {e}")
            return False

    def export_knowledge(self, filepath: str) -> bool:
        """
        Exports the knowledge base to a JSON file.

        Args:
            filepath (str): The path to the file where the knowledge base should be exported.

        Returns:
            bool: True if the export was successful, False otherwise.
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.storage, f, indent=4)
            logger.info(f"Knowledge base successfully exported to '{filepath}'.")
            return True
        except Exception as e:
            logger.error(f"Failed to export knowledge base to '{filepath}': {e}")
            return False