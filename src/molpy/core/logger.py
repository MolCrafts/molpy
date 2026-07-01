"""
Logging utilities for MolPy.

Thin wrapper over :mod:`mollog` (the molcrafts structured-logging library).
The mollog root logger is configured once, lazily, using MolPy's standard
format and the level from :data:`molpy.core.config.config`; named loggers then
propagate to it. mollog is a drop-in superset of the stdlib ``logging`` API, so
the returned loggers expose the familiar ``debug``/``info``/``warning``/
``error``/``critical`` methods.
"""

from __future__ import annotations

import threading

import mollog

from molpy.core.config import config

_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False
_config_lock = threading.Lock()


def _ensure_configured() -> None:
    """Configure the mollog root logger once, in a thread-safe way."""
    global _configured
    if _configured:
        return
    with _config_lock:
        if _configured:
            return
        mollog.configure(
            level=config.log_level,
            format=_LOG_FORMAT,
            capture_stdlib=False,
        )
        _configured = True


def get_logger(name: str) -> mollog.Logger:
    """
    Get a configured MolPy logger.

    Ensures the mollog root logger is configured with MolPy's standard
    timestamp + name + level + message format, returns the named logger (which
    propagates to the configured root), and sets its level from
    :data:`molpy.core.config.config` so ``Config.update(log_level=...)`` takes
    effect for loggers created afterwards.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        A :class:`mollog.Logger` instance.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        2024-01-01 12:00:00,000 - mymodule - INFO - Processing started
    """
    _ensure_configured()
    logger = mollog.get_logger(name)
    logger.set_level(config.log_level)
    return logger
