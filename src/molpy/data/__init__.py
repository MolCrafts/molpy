"""
MolPy Data Module - Access to built-in data files.

This module provides a unified interface for accessing built-in data files
such as force field parameters, molecule templates, and other resources.

Usage:
    from molpy.data import get_forcefield_path, get_path, list_files

    # Get path to a data file
    path = get_path("forcefield/tip3p.xml")

    # List available files in a subdirectory
    files = list_files("forcefield")

    # Get force field path (convenience function)
    ff_path = get_forcefield_path("tip3p.xml")
"""

from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path


def get_path(relative_path: str | Path) -> Path:
    """
    Get the absolute path to a data file.

    Args:
        relative_path: Relative path to the data file (e.g., "forcefield/tip3p.xml")

    Returns:
        Path object pointing to the data file

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from molpy.data import get_path
        >>> path = get_path("forcefield/tip3p.xml")
        >>> print(path)
        /path/to/molpy/data/forcefield/tip3p.xml
    """
    relative_path = Path(relative_path)

    resource = files(__package__ or "molpy.data")
    for part in relative_path.parts:
        resource = resource / part

    with as_file(resource) as path:
        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {relative_path}. "
                f"Available files: {list(list_files(relative_path.parent))}"
            )
        return path


def list_files(
    subdirectory: str | Path = "", exclude_python: bool = True
) -> Iterator[str]:
    """
    List all files in a data subdirectory.

    Args:
        subdirectory: Subdirectory to list (e.g., "forcefield"), empty string for root
        exclude_python: If True, exclude Python files (__init__.py, *.py, etc.)

    Yields:
        Relative paths to files in the subdirectory

    Examples:
        >>> from molpy.data import list_files
        >>> for file in list_files("forcefield"):
        ...     print(file)
        forcefield/tip3p.xml
        forcefield/tip3p.xml
    """
    subdirectory = Path(subdirectory)

    def _should_exclude(filename: str) -> bool:
        """Check if a file should be excluded."""
        if not exclude_python:
            return False
        # Exclude Python files
        return bool(filename.endswith(".py") or filename == "__pycache__")

    resource = files(__package__ or "molpy.data")
    for part in subdirectory.parts:
        resource = resource / part

    if resource.is_dir():
        for item in resource.iterdir():
            if item.is_file():
                if _should_exclude(item.name):
                    continue
                yield str(subdirectory / item.name)
            elif item.is_dir() and not _should_exclude(item.name):
                yield from list_files(
                    subdirectory / item.name, exclude_python=exclude_python
                )


def exists(relative_path: str | Path) -> bool:
    """
    Check if a data file exists.

    Args:
        relative_path: Relative path to the data file

    Returns:
        True if the file exists, False otherwise

    Examples:
        >>> from molpy.data import exists
        >>> if exists("forcefield/tip3p.xml"):
        ...     print("File exists")
    """
    try:
        get_path(relative_path)
        return True
    except FileNotFoundError:
        return False


# Convenience functions for specific data types
def get_forcefield_path(filename: str) -> Path:
    """
    Get the path to a force field file.

    Args:
        filename: Name of the force field file (e.g., "tip3p.xml")

    Returns:
        Path object pointing to the force field file

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from molpy.data import get_forcefield_path
        >>> path = get_forcefield_path("tip3p.xml")
        >>> print(path)
        /path/to/molpy/data/forcefield/tip3p.xml
    """
    return get_path(f"forcefield/{filename}")


def list_forcefields() -> list[str]:
    """
    List all available force field files.

    Returns:
        List of force field filenames

    Examples:
        >>> from molpy.data import list_forcefields
        >>> forcefields = list_forcefields()
        >>> print(forcefields)
        ["clp.xml", "tip3p.xml"]
    """
    return [Path(f).name for f in list_files("forcefield", exclude_python=True)]


__all__ = [
    "exists",
    "get_forcefield_path",
    "get_path",
    "list_files",
    "list_forcefields",
]
