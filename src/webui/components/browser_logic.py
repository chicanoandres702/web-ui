import os
import shutil
import logging
from typing import Optional, TYPE_CHECKING

from src.utils import config

if TYPE_CHECKING:
    from src.webui.webui_manager import WebuiManager

logger = logging.getLogger(__name__)

async def close_browser(webui_manager: "WebuiManager"):
    """
    Close the browser instance and context managed by WebuiManager.
    Cancels any running task associated with the browser.
    """
    if webui_manager.bu_current_task and not webui_manager.bu_current_task.done():
        webui_manager.bu_current_task.cancel()
        webui_manager.bu_current_task = None

    if webui_manager.bu_browser_context:
        logger.info("⚠️ Closing browser context when changing browser config.")
        await webui_manager.bu_browser_context.close()
        webui_manager.bu_browser_context = None

    if webui_manager.bu_browser:
        logger.info("⚠️ Closing browser when changing browser config.")
        await webui_manager.bu_browser.close()
        webui_manager.bu_browser = None

async def clear_browser_session(webui_manager: "WebuiManager", user_data_dir: str) -> str:
    """
    Clears the browser session directory.
    Closes the browser first to release file locks.
    """
    # Close browser first to release file locks
    await close_browser(webui_manager)
    
    path = user_data_dir if user_data_dir and user_data_dir.strip() else config.DEFAULT_BROWSER_SESSION_DIR
    abs_path = os.path.abspath(path)
    
    if os.path.exists(abs_path):
        try:
            # Basic safety check to prevent deleting root or short paths
            if len(abs_path.split(os.sep)) <= 2:
                 return f"❌ Safety Block: Path too short/root, refusing to delete: {abs_path}"
            
            shutil.rmtree(abs_path)
            os.makedirs(abs_path, exist_ok=True)
            return f"✅ Session cleared and recreated: {abs_path}"
        except Exception as e:
            return f"❌ Error clearing session: {e}"
    return f"⚠️ Session directory not found: {abs_path}"

def get_session_info(user_data_dir: str) -> str:
    """
    Checks the session directory size and existence.
    Returns a formatted string with status.
    """
    path = user_data_dir if user_data_dir and user_data_dir.strip() else config.DEFAULT_BROWSER_SESSION_DIR
    abs_path = os.path.abspath(path)
    
    if not os.path.exists(abs_path):
        return f"❌ Directory not found: {abs_path}\nSession data is likely not being saved here yet."
    
    total_size = 0
    file_count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(abs_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
                file_count += 1
    except Exception as e:
        return f"⚠️ Error reading directory: {e}"
    
    size_mb = total_size / (1024 * 1024)
    return f"✅ Session Directory Exists: {abs_path}\n📁 Files: {file_count} | 💾 Size: {size_mb:.2f} MB"

def update_env_var(key: str, value: Optional[str]):
    """Updates or removes an environment variable."""
    if value:
        os.environ[key] = str(value)
    else:
        os.environ.pop(key, None)
