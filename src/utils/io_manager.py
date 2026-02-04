import asyncio
import logging
import os
from typing import Optional, Any

logger = logging.getLogger(__name__)

class IOManager:
    """
    Streamlined I/O Manager for handling file operations asynchronously.
    Ensures the main event loop is not blocked by file system operations.
    """

    @staticmethod
    async def write_file(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> bool:
        """
        Asynchronously writes content to a file.
        """
        try:
            dir_path = os.path.dirname(os.path.abspath(path))
            
            def _write_op():
                os.makedirs(dir_path, exist_ok=True)
                with open(path, mode, encoding=encoding) as f:
                    f.write(content)
            
            await asyncio.to_thread(_write_op)
            return True
        except Exception as e:
            logger.error(f"IOManager Write Error ({path}): {e}")
            return False

    @staticmethod
    async def read_file(path: str, encoding: str = "utf-8") -> Optional[str]:
        """
        Asynchronously reads content from a file.
        """
        if not os.path.exists(path):
            return None
            
        try:
            def _read_op():
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            
            return await asyncio.to_thread(_read_op)
        except Exception as e:
            logger.error(f"IOManager Read Error ({path}): {e}")
            return None

    @staticmethod
    def write_file_sync(path: str, content: str, mode: str = "w", encoding: str = "utf-8") -> bool:
        """
        Synchronous fallback for non-async contexts.
        """
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"IOManager Sync Write Error ({path}): {e}")
            return False

    @staticmethod
    def read_file_sync(path: str, encoding: str = "utf-8") -> Optional[str]:
        """
        Synchronous fallback for reading files.
        """
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except Exception as e:
            logger.error(f"IOManager Sync Read Error ({path}): {e}")
            return None
