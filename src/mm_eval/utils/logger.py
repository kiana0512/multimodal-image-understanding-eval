"""Logging helpers."""
from __future__ import annotations
import logging

def get_logger(name: str = "mm_eval") -> logging.Logger:
    """Return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    try:
        from rich.logging import RichHandler
        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
    except Exception:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
