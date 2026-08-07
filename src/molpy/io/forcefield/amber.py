"""AMBER prmtop I/O — thin molrs façade.

* Structure: :func:`molrs.io.read_amber_prmtop`
* Force field: :func:`molrs.ff.read_amber_prmtop_ff` (LAMMPS form map)
* Table decode (POINTERS, bonds, angles, dihedrals, LJ, 20a4 names):
  :mod:`molrs.io` ``prmtop_*`` helpers — call those directly; this module
  does **not** re-export or wrap them.
"""

from __future__ import annotations

from pathlib import Path

from molrs import Frame

# AMBER FileFormats: charges × √(332.0636) ≈ 18.2223 → electron units in Frame.
CHARGE_CONVERSION_FACTOR = 18.2223


class AmberPrmtopReader:
    """Read AMBER prmtop structure + force field via molrs.

    Args:
        file: Path to a ``.prmtop`` / ``.parm7`` file.
    """

    def __init__(self, file: str | Path):
        self.file = Path(file)

    def read(self, frame: Frame | None = None) -> tuple[Frame, object]:
        """Load structure and force field.

        Args:
            frame: Optional destination Frame; when given, structure blocks and
                meta are copied into it. When ``None``, the molrs Frame is
                returned as-is.

        Returns:
            ``(frame, forcefield)``

        Raises:
            FileNotFoundError: Missing path.
            ValueError: Invalid / empty prmtop or FF parse failure.
        """
        path = self.file
        if not path.is_file():
            raise FileNotFoundError(path)

        import molrs.ff as mff
        import molrs.io

        try:
            structure = molrs.io.read_amber_prmtop(str(path))
        except OSError as exc:
            msg = str(exc)
            if "POINTERS" in msg:
                raise ValueError(
                    f"Invalid or empty prmtop file '{self.file}': POINTERS section missing. "
                    "This typically means the external tool (tleap) failed to create the file."
                ) from exc
            raise ValueError(msg) from exc

        try:
            ff = mff.read_amber_prmtop_ff(str(path))
        except Exception as exc:
            raise ValueError(f"Failed to parse prmtop force field: {exc}") from exc

        if frame is None:
            return structure, ff

        for key in ("atoms", "bonds", "angles", "dihedrals", "impropers"):
            if key in structure:
                frame[key] = structure[key]
        frame.meta = {**frame.meta, **dict(structure.meta)}
        return frame, ff
