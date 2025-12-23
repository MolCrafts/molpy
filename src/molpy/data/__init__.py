"""
MolPy Data Module - Access to built-in data files.

This module provides a unified interface for accessing built-in data files
such as force field parameters, molecule templates, and other resources.

Usage:
    from molpy.data import get_path, list_files
    from molpy.data.forcefield import get_forcefield_path

    # Get path to a data file
    path = get_path("forcefield/oplsaa.xml")

    # List available files in a subdirectory
    files = list_files("forcefield")

    # Get force field path (convenience function)
    ff_path = get_forcefield_path("oplsaa.xml")
"""

from collections.abc import Iterator
from importlib.resources import as_file, files
from importlib.resources.abc import Traversable
from pathlib import Path


def _get_data_root() -> Path:
    """Get the root directory of the data package."""
    try:
        return Path(files(__package__ or "molpy.data"))
    except Exception:
        # Fallback to file-based approach
        return Path(__file__).parent


def get_path(relative_path: str | Path) -> Path:
    """
    Get the absolute path to a data file.

    Args:
        relative_path: Relative path to the data file (e.g., "forcefield/oplsaa.xml")

    Returns:
        Path object pointing to the data file

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from molpy.data import get_path
        >>> path = get_path("forcefield/oplsaa.xml")
        >>> print(path)
        /path/to/molpy/data/forcefield/oplsaa.xml
    """
    relative_path = Path(relative_path)

    try:
        resource = files(__package__ or "molpy.data")
        # Navigate to the resource
        for part in relative_path.parts:
            resource = resource / part

        # Convert to file path
        with as_file(resource) as path:
            if path.exists():
                return path
            else:
                raise FileNotFoundError(
                    f"Data file not found: {relative_path}. "
                    f"Available files: {list(list_files(relative_path.parent))}"
                )
    except Exception as e:
        # Fallback to file-based approach
        data_root = _get_data_root()
        full_path = data_root / relative_path
        if full_path.exists():
            return full_path
        else:
            raise FileNotFoundError(
                f"Data file not found: {relative_path}. Checked: {full_path}"
            ) from e


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
        forcefield/oplsaa.xml
        forcefield/tip3p.xml
    """
    subdirectory = Path(subdirectory)

    def _should_exclude(filename: str) -> bool:
        """Check if a file should be excluded."""
        if not exclude_python:
            return False
        # Exclude Python files
        return bool(filename.endswith(".py") or filename == "__pycache__")

    try:
        resource = files(__package__ or "molpy.data")
        # Navigate to the subdirectory
        for part in subdirectory.parts:
            resource = resource / part

        # List files
        if resource.is_dir():
            for item in resource.iterdir():
                if item.is_file():
                    # Skip Python files if requested
                    if _should_exclude(item.name):
                        continue
                    # Return relative path
                    rel_path = subdirectory / item.name
                    yield str(rel_path)
                elif item.is_dir() and not _should_exclude(item.name):
                    # Recursively list files in subdirectories
                    rel_path = subdirectory / item.name
                    yield from list_files(rel_path, exclude_python=exclude_python)
    except Exception:
        # Fallback to file-based approach
        data_root = _get_data_root()
        subdir_path = data_root / subdirectory
        if subdir_path.exists() and subdir_path.is_dir():
            for item in subdir_path.iterdir():
                if item.is_file():
                    # Skip Python files if requested
                    if _should_exclude(item.name):
                        continue
                    rel_path = subdirectory / item.name
                    yield str(rel_path)
                elif item.is_dir() and not _should_exclude(item.name):
                    rel_path = subdirectory / item.name
                    yield from list_files(rel_path, exclude_python=exclude_python)


def exists(relative_path: str | Path) -> bool:
    """
    Check if a data file exists.

    Args:
        relative_path: Relative path to the data file

    Returns:
        True if the file exists, False otherwise

    Examples:
        >>> from molpy.data import exists
        >>> if exists("forcefield/oplsaa.xml"):
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
        filename: Name of the force field file (e.g., "oplsaa.xml")

    Returns:
        Path object pointing to the force field file

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from molpy.data import get_forcefield_path
        >>> path = get_forcefield_path("oplsaa.xml")
        >>> print(path)
        /path/to/molpy/data/forcefield/oplsaa.xml
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
        ['oplsaa.xml', 'tip3p.xml']
    """
    return [Path(f).name for f in list_files("forcefield", exclude_python=True)]


__all__ = [
    "exists",
    "get_forcefield_path",
    "get_path",
    "list_files",
    "list_forcefields",
]
