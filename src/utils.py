"""
Utility functions for logging, file operations, and general helpers.
"""

import os
import re
import datetime
from pathlib import Path

from .config import LOGS_DIR


def get_current_ts_string() -> str:
    """Get current timestamp as a formatted string."""
    now = datetime.datetime.now()
    return now.strftime("%H:%M:%S.%f")[:-3]


def log(msg: str) -> None:
    """Print a timestamped log message."""
    print(f"[{get_current_ts_string()}] {msg}", flush=True)


def get_log_path(filename: str) -> str:
    """Get full path for a log file in the logs directory."""
    return str(LOGS_DIR / filename)


def sanitize_filename(name: str) -> str:
    """Make a safe ASCII filename segment from a model or label."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name or "")
    cleaned = cleaned.strip("._")
    return cleaned or "unnamed"


def append_to_file(filepath: str, text: str) -> None:
    """Append text to a file (best-effort, silently fails)."""
    if not text:
        return
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception as e:
        log(f"FILE ERROR ({filepath}): {e}")
