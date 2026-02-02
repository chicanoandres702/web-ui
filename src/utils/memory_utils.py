import os
import logging
import shutil

logger = logging.getLogger(__name__)

def configure_mem0():
    """
    Configures environment variables for Mem0 to ensure it runs locally and without errors.
    """
    try:
        # Set local storage directory to avoid permission issues or clutter
        mem0_dir = os.path.abspath("./tmp/mem0_storage")
        os.makedirs(mem0_dir, exist_ok=True)
        
        # Only set if not already set, to allow user override via .env
        if "MEM0_DIR" not in os.environ:
            os.environ["MEM0_DIR"] = mem0_dir
            # logger.info(f"Mem0 configured to use directory: {mem0_dir}")
        
        # Disable telemetry
        if "MEM0_TELEMETRY" not in os.environ:
            os.environ["MEM0_TELEMETRY"] = "false"
            
    except Exception as e:
        logger.error(f"Error configuring Mem0: {e}")

def reset_mem0():
    """Resets the Mem0 storage."""
    try:
        mem0_dir = os.environ.get("MEM0_DIR", os.path.abspath("./tmp/mem0_storage"))
        if os.path.exists(mem0_dir):
            shutil.rmtree(mem0_dir)
            os.makedirs(mem0_dir, exist_ok=True)
            logger.info("Mem0 storage reset.")
    except Exception as e:
        logger.error(f"Error resetting Mem0: {e}")