import logging

logger = logging.getLogger(__name__)

async def after_navigation_checks(agent):
    """
    Performs checks after navigation, including heuristics and subtask handling.
    """
    try:
        # Heuristic check for completion (e.g. Quiz done)
        await agent.heuristics.check_completion_heuristics()

        # Check for progress indicators
        await agent.heuristics.detect_progress()

    except Exception as e:
        logger.error(f"Error during post-navigation checks: {e}")