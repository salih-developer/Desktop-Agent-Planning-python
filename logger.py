"""Central logging configuration for Desktop Agent.

Log file: ~/.desktop_agent/agent.log
Rotation:  5 MB max, 3 backups kept
Level:     DEBUG (file) / WARNING (console)

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("message")
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_PATH = Path.home() / ".desktop_agent" / "agent.log"
_INITIALIZED = False


def setup_logging() -> None:
    """Call once at application startup (main.py)."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # File handler — rotating, UTF-8, full detail
    fh = logging.handlers.RotatingFileHandler(
        _LOG_PATH,
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — warnings+ only (don't pollute terminal)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root.addHandler(fh)
    root.addHandler(ch)

    # Silence noisy third-party libraries
    for lib in ("httpx", "httpcore", "urllib3", "ollama"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. setup_logging() must be called first."""
    return logging.getLogger(name)
