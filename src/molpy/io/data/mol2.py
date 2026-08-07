"""Tripos MOL2 structure I/O (molrs-backed).

Read/write go through :func:`molrs.io.read_mol2` / :func:`molrs.io.write_mol2`.
Canonical column names: ``type`` (SYBYL atom type), ``res_id``/``res_name``
(from ``subst_*``).
"""

from __future__ import annotations

from pathlib import Path

from molrs import Frame

from .base import DataReader, DataWriter


class Mol2Reader(DataReader):
    """Read a Tripos MOL2 file into a :class:`~molpy.Frame` (first molecule)."""

    def __init__(self, file: str | Path) -> None:
        super().__init__(Path(file))
        self._file = Path(file)

    def read(self, frame: Frame | None = None) -> Frame:
        """Read the first molecule from the MOL2 path.

        Args:
            frame: Accepted for API parity; ignored (molrs always returns a
                new Frame).

        Returns:
            Canonical Frame (``type``, ``res_id``, ``res_name``, …).
        """
        del frame
        import molrs.io

        if not self._file.exists():
            raise FileNotFoundError(f"MOL2 file not found: {self._file}")
        try:
            return molrs.io.read_mol2(str(self._file))
        except OSError as exc:
            msg = str(exc).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(str(exc)) from exc
            raise ValueError(f"Failed to read MOL2 file: {exc}") from exc


class Mol2Writer(DataWriter):
    """Write a Frame to a Tripos MOL2 file (molrs)."""

    def __init__(self, file: str | Path) -> None:
        super().__init__(Path(file))
        self._file = Path(file)

    def write(self, frame: Frame) -> None:
        """Write *frame* as MOL2 (canonical columns)."""
        import molrs.io

        molrs.io.write_mol2(str(self._file), frame)
