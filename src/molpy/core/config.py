"""
Global configuration system for MolPy.

Thin singleton over :class:`molcfg.Config` (the molcrafts configuration
library). Configuration can be read globally, updated, reset, or temporarily
overridden with a context manager. Because the singleton is mutated in place,
the module-level :data:`config` reference always reflects the current values.

Examples:
    >>> from molpy.core.config import config, Config
    >>>
    >>> # Access current config
    >>> print(config.log_level)
    INFO
    >>>
    >>> # Update config globally
    >>> Config.update(log_level="DEBUG", n_threads=4)
    >>> print(config.n_threads)
    4
    >>>
    >>> # Temporary override
    >>> with Config.temporary(log_level="WARNING"):
    ...     print(config.log_level)
    WARNING
    >>> print(config.log_level)
    DEBUG
    >>> Config.reset()
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from typing import Any

from molcfg import Config as _MolcfgConfig

# Default configuration values. ``reset()`` restores exactly these keys.
_DEFAULTS: dict[str, Any] = {
    "log_level": "INFO",
    "n_threads": 1,
}


class Config(_MolcfgConfig):
    """
    Global configuration for MolPy, backed by :class:`molcfg.Config`.

    Thread-safe singleton storing global settings such as the logging level
    and parallelization parameters. Use the class methods to access and modify
    the shared instance; values are read through attribute access
    (``config.log_level``) or dotted-path access (``config["log_level"]``).

    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        n_threads: Number of threads for parallel computations.
    """

    _instance: Config | None = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> Config:
        """
        Get the singleton Config instance.

        Thread-safe lazy initialization. Creates the instance on first call
        seeded with the default values.

        Returns:
            The singleton Config instance.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(dict(_DEFAULTS))
            return cls._instance

    @classmethod
    def update(cls, **kwargs: Any) -> None:
        """
        Update the global configuration in place.

        Thread-safe update. Changes persist until :meth:`reset` or another
        :meth:`update`.

        Args:
            **kwargs: Configuration fields to update (log_level, n_threads, ...).
        """
        inst = cls.instance()
        with cls._lock:
            for key, value in kwargs.items():
                setattr(inst, key, value)

    @classmethod
    def reset(cls) -> None:
        """
        Reset configuration to default values in place.

        Thread-safe reset. Removes any keys added at runtime and restores the
        documented defaults.
        """
        inst = cls.instance()
        with cls._lock:
            for key in list(inst.keys()):
                delattr(inst, key)
            for key, value in _DEFAULTS.items():
                setattr(inst, key, value)

    @classmethod
    @contextlib.contextmanager
    def temporary(cls, **overrides: Any) -> Generator[None, None, None]:
        """
        Temporarily override configuration within a context.

        Thread-safe context manager that snapshots the current state on entry
        and restores it on exit. Useful for testing or scoped parameter changes.

        Args:
            **overrides: Configuration fields to temporarily override.

        Yields:
            None
        """
        inst = cls.instance()
        with cls._lock:
            inst.snapshot()
            for key, value in overrides.items():
                setattr(inst, key, value)
        try:
            yield
        finally:
            with cls._lock:
                inst.rollback()

    def __repr__(self) -> str:
        return f"<MolpyConfig: {self.to_dict()}>"


config = Config.instance()
"""Global config instance. Use this for read access."""


def get_config() -> Config:
    """
    Get the global configuration instance.

    Convenience function equivalent to :meth:`Config.instance`.

    Returns:
        The singleton Config instance.
    """
    return config
