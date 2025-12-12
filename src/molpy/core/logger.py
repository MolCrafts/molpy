"""
Logging utilities for MolPy.

Provides configured logger instances with consistent formatting.
"""
import logging


def get_logger(name: str) -> logging.Logger:
    """
    Create a configured logger instance.
    
    Creates a logger with DEBUG level and stream handler using
    a standard timestamp + name + level + message format.
    
    Args:
        name: Logger name (typically __name__ of calling module)
    
    Returns:
        Configured logging.Logger instance
    
    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        2024-01-01 12:00:00,000 - mymodule - INFO - Processing started
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger

