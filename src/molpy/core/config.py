"""
Global configuration system for MolPy.

This module provides a thread-safe singleton configuration system using Pydantic.
Configuration can be accessed globally, updated, reset, or temporarily overridden
using a context manager.

Examples:
    >>> from molpy.core.config import config, Config
    >>>
    >>> # Access current config
    >>> print(config.log_level)
    'INFO'
    >>>
    >>> # Update config globally
    >>> Config.update(log_level='DEBUG', n_threads=4)
    >>>
    >>> # Temporary override
    >>> with Config.temporary(log_level='WARNING'):
    ...     print(config.log_level)  # 'WARNING'
    >>> print(config.log_level)  # Back to 'DEBUG'
"""

import contextlib
import threading
from collections.abc import Generator
from typing import Self

from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Global configuration for MolPy.

    Thread-safe singleton that stores global settings like logging level
    and parallelization parameters. Use class methods to access and modify
    the singleton instance.

    Attributes:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        n_threads: Number of threads for parallel computations

    Examples:
        >>> # Get singleton instance
        >>> cfg = Config.instance()
        >>>
        >>> # Update configuration
        >>> Config.update(n_threads=8)
        >>>
        >>> # Temporary override
        >>> with Config.temporary(log_level='DEBUG'):
        ...     # Config is DEBUG here
        ...     pass
        >>> # Config restored here
    """

    log_level: str = Field(default="INFO")

    n_threads: int = Field(
        default=1, description="Number of threads to use for parallel computations"
    )

    # --- Internal singleton & lock ---
    _instance: Self = None
    _lock = threading.Lock()

    # --- Singleton accessor ---
    @classmethod
    def instance(cls) -> Self:
        """
        Get the singleton Config instance.

        Thread-safe lazy initialization. Creates instance on first call.

        Returns:
            The singleton Config instance

        Examples:
            >>> cfg = Config.instance()
            >>> print(cfg.log_level)
            'INFO'
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # --- Update config ---
    @classmethod
    def update(cls, **kwargs):
        """
        Update the global configuration.

        Thread-safe update that creates a new instance with updated values.
        Changes persist until reset() or another update().

        Args:
            **kwargs: Configuration fields to update (log_level, n_threads, etc.)

        Examples:
            >>> Config.update(log_level='DEBUG', n_threads=4)
            >>> cfg = Config.instance()
            >>> print(cfg.n_threads)
            4
        """
        with cls._lock:
            cls._instance = cls.instance().model_copy(update=kwargs)

    # --- Reset to default ---
    @classmethod
    def reset(cls):
        """
        Reset configuration to default values.

        Thread-safe reset. Creates a new instance with all default values.

        Examples:
            >>> Config.update(n_threads=8)
            >>> Config.reset()
            >>> print(Config.instance().n_threads)
            1
        """
        with cls._lock:
            cls._instance = cls()

    # --- Context manager for temporary override ---
    @classmethod
    @contextlib.contextmanager
    def temporary(cls, **overrides) -> Generator[None, None, None]:
        """
        Temporarily override configuration within a context.

        Thread-safe context manager that restores original config on exit.
        Useful for testing or temporary parameter changes.

        Args:
            **overrides: Configuration fields to temporarily override

        Yields:
            None

        Examples:
            >>> original_level = Config.instance().log_level
            >>> with Config.temporary(log_level='DEBUG'):
            ...     print(Config.instance().log_level)  # 'DEBUG'
            >>> print(Config.instance().log_level)  # original_level
        """
        with cls._lock:
            old = cls.instance().model_copy()
            cls._instance = cls._instance.model_copy(update=overrides)
        try:
            yield
        finally:
            with cls._lock:
                cls._instance = old

    def __repr__(self) -> str:
        return f"<MolpyConfig: {self}>"


config = Config.instance()
"""Global config instance. Use this for read access."""


def get_config() -> Config:
    """
    Get the global configuration instance.

    Convenience function equivalent to Config.instance().

    Returns:
        The singleton Config instance

    Examples:
        >>> cfg = get_config()
        >>> print(cfg.log_level)
        'INFO'
    """
    return config
