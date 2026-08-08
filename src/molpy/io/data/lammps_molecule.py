"""LAMMPS molecule template I/O — molrs-backed.

Native ``.mol`` and JSON molecule files are read/written by
:func:`molrs.io.read_lammps_molecule` /
:func:`molrs.io.write_lammps_molecule`. This module is a thin façade that
keeps the historical class API and applies the LAMMPS field formatter on
read (``q`` → ``charge`` when present).
"""

from __future__ import annotations

from pathlib import Path

from molrs import Frame

from molpy.io.data.lammps import LammpsFieldFormatter

from .base import DataReader, DataWriter


class LammpsMoleculeReader(DataReader):
    """LAMMPS molecule file reader (native or JSON by suffix)."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(Path(path))

    _formatter = LammpsFieldFormatter()

    def read(self, frame: Frame | None = None) -> Frame:
        """Read a molecule template into a Frame."""
        del frame
        import molrs.io

        if not self._path.exists():
            raise FileNotFoundError(f"Molecule file not found: {self._path}")
        try:
            out = molrs.io.read_lammps_molecule(str(self._path))
        except OSError as e:
            msg = str(e)
            if "Empty" in msg or "empty" in msg:
                raise ValueError(msg) from e
            if "Types" in msg or "format" in msg or "types" in msg:
                raise ValueError(msg) from e
            raise ValueError(msg) from e
        self._formatter.canonicalize_frame(out)
        return out


class LammpsMoleculeWriter(DataWriter):
    """LAMMPS molecule file writer (native or JSON)."""

    def __init__(self, path: str | Path, format_type: str = "native") -> None:
        super().__init__(Path(path))
        self.format_type = format_type.lower()
        if self.format_type not in ("native", "json"):
            raise ValueError("format_type must be 'native' or 'json'")
        if self.format_type == "json" and self._path.suffix.lower() != ".json":
            self._path = self._path.with_suffix(".json")

    def write(self, frame: Frame) -> None:
        """Write Frame to a LAMMPS molecule file."""
        import molrs.io

        if "atoms" not in frame:
            raise ValueError("Frame must contain atoms data")
        molrs.io.write_lammps_molecule(str(self._path), frame, format=self.format_type)
