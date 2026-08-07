"""GROMACS topology force-field I/O (molrs-backed).

Reader/writer unit conversion lives in
:func:`molrs.ff.read_gromacs_top_ff` / :func:`molrs.ff.write_gromacs_top_ff`.
Structure-only topology is :mod:`molpy.io.data.top`.
"""

from __future__ import annotations

from pathlib import Path

from molpy.core.forcefield import ForceField


class GromacsTopReader:
    """Read a GROMACS ``.top`` / ``.itp`` into a :class:`~molpy.ForceField`.

    Structure tables are handled by :mod:`molpy.io.data.top` /
    :func:`molrs.io.read_top`. This class owns the force-field half via
    :func:`molrs.ff.read_gromacs_top_ff` (unit normalization at the boundary).
    """

    def __init__(self, file: str | Path, include: bool = False):
        self.file = Path(file)
        self.include = include

    def read(
        self,
        forcefield: ForceField | None = None,
        *,
        strip_comments: bool = True,
        recursive: bool = True,
    ) -> ForceField:
        """Parse the topology file into a ForceField."""
        del forcefield, strip_comments  # API parity only
        import molrs.ff as mff

        if not self.file.is_file():
            raise FileNotFoundError(self.file)

        try:
            return mff.read_gromacs_top_ff(
                str(self.file), include=bool(self.include and recursive)
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            msg = str(exc).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(str(exc)) from exc
            raise ValueError(f"Failed to read GROMACS top force field: {exc}") from exc


class GromacsForceFieldWriter:
    """Write a ForceField to GROMACS ``.top`` / ``.itp`` (via molrs)."""

    def __init__(self, filepath: str | Path, precision: int = 6) -> None:
        self._file = Path(filepath)
        self._prec = precision

    def write(self, forcefield: ForceField) -> None:
        """Serialize *forcefield* to GROMACS topology format."""
        import molrs.ff as mff

        try:
            mff.write_gromacs_top_ff(str(self._file), forcefield, precision=self._prec)
        except Exception as exc:
            raise ValueError(f"Failed to write GROMACS top force field: {exc}") from exc
