"""
Force field data files.

This submodule provides convenient access to force field data files.
"""

from molpy.data import get_path, list_files

__all__ = ["get_forcefield_path", "list_forcefields"]


def get_forcefield_path(filename: str) -> str:
    """
    Get the path to a force field file.

    Args:
        filename: Name of the force field file (e.g., "oplsaa.xml")

    Returns:
        Path string to the force field file

    Raises:
        FileNotFoundError: If the file does not exist

    Examples:
        >>> from molpy.data.forcefield import get_forcefield_path
        >>> path = get_forcefield_path("oplsaa.xml")
    """
    from pathlib import Path

    return str(get_path(f"forcefield/{filename}"))


def list_forcefields() -> list[str]:
    """
    List all available force field files.

    Returns:
        List of force field filenames

    Examples:
        >>> from molpy.data.forcefield import list_forcefields
        >>> forcefields = list_forcefields()
        >>> print(forcefields)
        ['oplsaa.xml', 'tip3p.xml']
    """
    from pathlib import Path

    return [Path(f).name for f in list_files("forcefield", exclude_python=True)]
