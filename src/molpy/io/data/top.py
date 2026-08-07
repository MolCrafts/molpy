"""GROMACS topology structure I/O (molrs-backed).

Read/write go through :func:`molrs.io.read_top` / :func:`molrs.io.write_top`.
Structure only (``[ atoms ]``, bonds/pairs/angles/dihedrals) — not force-field
parameter tables (see :mod:`molpy.io.forcefield.top`).

Connectivity atom indices are **1-based** as written in the file.
"""

from __future__ import annotations

from pathlib import Path

from molrs import Frame

from .base import DataReader, DataWriter, PathLike


class TopReader(DataReader):
    """Read GROMACS topology structure into a :class:`~molpy.Frame`.

    Examples:
        >>> reader = TopReader("molecule.top")
        >>> frame = reader.read()
        >>> frame["atoms"]  # Block with atom data
    """

    def __init__(self, file: PathLike, **open_kwargs):
        """Initialize GROMACS topology reader.

        Args:
            file: Path to GROMACS .top file
            **open_kwargs: Accepted for API parity; unused (molrs opens the path).
        """
        super().__init__(file, **open_kwargs)
        self._file = Path(file)

    def read(self, frame: Frame | None = None) -> Frame:
        """Read GROMACS topology file.

        Args:
            frame: Accepted for API parity; ignored (molrs always returns a
                new Frame).

        Returns:
            Frame with atoms and any connectivity blocks present.
        """
        del frame
        import molrs.io

        if not self._file.exists():
            raise FileNotFoundError(f"TOP file not found: {self._file}")
        try:
            return molrs.io.read_top(str(self._file))
        except OSError as exc:
            msg = str(exc).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(str(exc)) from exc
            raise ValueError(f"Failed to read TOP file: {exc}") from exc


class TopWriter(DataWriter):
    """Write a Frame as a minimal GROMACS topology structure file (molrs)."""

    def __init__(self, file: PathLike, **open_kwargs):
        super().__init__(file, **open_kwargs)
        self._file = Path(file)

    def write(self, frame: Frame) -> None:
        """Write *frame* as topology structure (canonical top columns)."""
        import molrs.io

        molrs.io.write_top(str(self._file), frame)
