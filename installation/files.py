"""
This module aggregates all file definitions for the project setup.
"""

# ==========================================
#        PROJECT FILE DEFINITIONS
# ==========================================
from installation.components.backend import CORE_BACKEND_FILES
from installation.components.auth import AUTH_FILES
from installation.components.browser import BROWSER_FILES
from installation.components.frontend import FRONTEND_FILES
from installation.components.config import CONFIG_FILES

# ==========================================
#        AGGREGATED FILES
# ==========================================
FILES = {
    **CORE_BACKEND_FILES,
    **AUTH_FILES,
    **BROWSER_FILES,
    **FRONTEND_FILES,
    **CONFIG_FILES,
}
