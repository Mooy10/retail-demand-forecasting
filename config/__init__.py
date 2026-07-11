"""Project configuration compatibility package.

This package exists because Phase 9 stores business assumption files under
`config/`. It re-exports the Python settings from `src.config` so legacy imports
such as `from config import PROCESSED_DATA_DIR` keep working.
"""

from src.config import *  # noqa: F401,F403
