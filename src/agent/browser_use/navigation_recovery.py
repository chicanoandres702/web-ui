"""
Specialized Assessment Logic for Navigation Recovery.
This module helps the agent recover when it becomes 'confused' by site layouts.
"""
from typing import Dict, List, Any, Union

def evaluate_site_state(page_content: str, current_step_index: int, total_steps: int) -> Dict[str, Any]:
    """
    Analyzes the current page content to determine if the agent is off-track.
    Args:
        page_content: Combined text content of the page (and optionally URL).
        current_step_index: Current step number.
        total_steps: Max steps allowed.
    """
    content_lower = page_content.lower()
    
    # Logic to detect if we are trapped in an ad or vignette
    is_vignette = "google_vignette" in content_lower or "dismiss" in content_lower
    
    # Logic to find the quiz marker
    has_quiz_marker = "question" in content_lower and "of" in content_lower
    
    # Logic to check if logged in (prevents false 'lost' positives on dashboard)
    is_logged_in = any(x in content_lower for x in ["log out", "sign out", "my account", "profile", "dashboard"])
    
    # Check for empty/blank page
    is_blank = "about:blank" in content_lower or not content_lower.strip()
    
    # Check for common error pages
    is_error_page = any(x in content_lower for x in ["404 not found", "500 internal server error", "access denied", "site can't be reached", "dns_probe_finished", "bad gateway", "service unavailable"])
    is_captcha = "captcha" in content_lower or "verify you are human" in content_lower or "security check" in content_lower

    assessment = {
        "status": "on_track",
        "action_required": None,
        "reason": ""
    }

    if is_blank:
        assessment["status"] = "lost"
        assessment["action_required"] = "RENAVIGATE"
        assessment["reason"] = "Browser is on about:blank. Navigation likely failed."
    elif is_error_page:
        assessment["status"] = "failed"
        assessment["action_required"] = "GO_BACK_AND_RETRY"
        assessment["reason"] = "Browser encountered a critical error page (404/500/Access Denied)."
    elif is_captcha:
        assessment["status"] = "blocked"
        assessment["action_required"] = "SOLVE_CAPTCHA"
        assessment["reason"] = "CAPTCHA detected."
    elif is_vignette:
        assessment["status"] = "blocked"
        assessment["action_required"] = "CLOSE_OVERLAY"
        assessment["reason"] = "Detected Google Vignette or overlay."
    elif "authenticator.pingone.com" in content_lower:
        assessment["status"] = "mfa_required"
        assessment["action_required"] = "HANDLE_MFA"
    elif not has_quiz_marker and not is_logged_in:
        # Only consider it 'lost' if it's not the very beginning of the task
        if current_step_index > 3: # Heuristic: after initial navigation and clearing
            assessment["status"] = "lost"
            assessment["action_required"] = "SCROLL_TO_ANCHOR"
            assessment["reason"] = "Lost visual contact with quiz container. Re-orienting..."
    
    return assessment

def generate_recovery_plan(failure_log: Union[List[Any], str]) -> List[Dict[str, str]]:
    """
    Takes the 'confused' logs and generates a micro-plan to get back to the quiz.
    """
    return [
        {"step": "Reload the page to clear temporary state", "action": "reload"},
        {"step": "Use clear_view to remove any new overlays", "action": "clear_view"},
        {"step": "Search for 'Question 1' to find the start point", "action": "scroll_to_text", "args": {"text": "Question 1"}}
    ]
