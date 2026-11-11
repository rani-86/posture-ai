"""Centralized logger configuration so every module logs consistently."""
from __future__ import annotations

import sys

from loguru import logger

_CONFIGURED = False


def get_logger(name: str):
    """Return the shared loguru logger, configured once per process."""
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{extra[module]}</cyan> - {message}",
            level="INFO",
        )
        _CONFIGURED = True
    return logger.bind(module=name)
