"""IO utility functions.

This module intentionally keeps only utilities that are valid in the current
MolPy core layout.
"""

from __future__ import annotations

from pathlib import Path

from molpy.core.frame import Frame


class ZipReader:
    """
    Zip multiple readers together for parallel iteration.

    Context manager that yields tuples of frames from multiple readers.

    Args:
        *readers: Variable number of reader objects
    """

    def __init__(self, *readers):
        self.readers = readers

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close all readers."""
        for reader in self.readers:
            reader.__exit__(exc_type, exc_val, exc_tb)

    def __iter__(self):
        """Iterate over zipped frames from all readers."""
        yield from zip(*self.readers)


def ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory for `path` exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
