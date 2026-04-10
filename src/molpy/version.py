"""
Version information for MolPy.

This module provides simple version information for MolPy.
"""

version = "0.3.0"
release_date = "2026-04-08"


def __str__() -> str:
    """String representation of version."""
    return version


def __repr__() -> str:
    """Detailed string representation of version."""
    return f"MolPy version {version} (released {release_date})"


# Export version attributes
__all__ = [
    "version",
    "release_date",
]
