"""Structured logging configuration for the ResearchLens application.

Provides a factory function that returns a consistently-configured logger
so that every module shares the same format and level.
"""

import logging
import os
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_LEVEL = "INFO"

# Track whether the root handler has already been attached to avoid
# duplicate log lines when ``setup_logger`` is called from many modules.
_root_handler_installed: bool = False


def setup_logger(name: str) -> logging.Logger:
    """Create (or retrieve) a logger with the ResearchLens conventions.

    Parameters
    ----------
    name:
        Dot-separated logger name, typically ``__name__`` of the caller.

    Returns
    -------
    logging.Logger
        A logger configured with the standard format and a ``StreamHandler``
        writing to *stdout*.
    """
    global _root_handler_installed  # noqa: PLW0603

    level_name = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Attach a shared handler to the root logger only once so that child
    # loggers inherit it without producing duplicate lines.
    if not _root_handler_installed:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(level)
        _root_handler_installed = True

    return logger
