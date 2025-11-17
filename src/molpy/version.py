"""
Version information for MolPy.

This module provides simple version information for MolPy.
"""

# Version information
__version__ = "0.2.0"
__release_date__ = "2025-11-17"

# NumPy-style version attributes for compatibility
version = __version__


def __str__() -> str:
    """String representation of version."""
    return __version__


def __repr__() -> str:
    """Detailed string representation of version."""
    return f"MolPy version {__version__} (released {__release_date__})"


# Export version attributes
__all__ = [
    "version",
]
