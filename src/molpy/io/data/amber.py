from pathlib import Path
from typing import Tuple

import numpy as np
import molpy as mp

from molpy.core import Frame, Block
from .base import DataReader

# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _read_fortran_12(data_str: str) -> np.ndarray:
    """Split a concatenated 12-column string → float array."""
    vals = [float(data_str[i : i + 12]) for i in range(0, len(data_str), 12)]
    return np.asarray(vals, dtype=float)


def _eat_section(lines: list[str], n_values: int) -> Tuple[np.ndarray, int]:
    """
    Consume enough lines to collect *n_values* 12-column floats.

    Returns
    -------
    array, int
        Parsed floats and number of lines consumed.
    """
    floats: list[float] = []
    consumed = 0
    while len(floats) < n_values:
        floats.extend(_read_fortran_12(lines[consumed]))
        consumed += 1
    return np.asarray(floats[:n_values]), consumed


# ---------------------------------------------------------------------
# main reader
# ---------------------------------------------------------------------
class AmberInpcrdReader(DataReader):
    """
    Reader for AMBER ASCII `*.inpcrd` (old-style) coordinate files.

    * Coordinates: 12.7/12.8 format, 6 numbers per line
    * Optional velocities section (same length as coordinates)
    * Optional final box line (3–6 floats)
    """

    __slots__ = ()

    def __init__(self, file: str | Path, **kwargs):
        super().__init__(path=Path(file), **kwargs)

    # ------------------------------------------------------------------
    def read(self, frame: mp.Frame | None = None) -> mp.Frame:
        frame = frame or mp.Frame()

        raw_lines = self.read_lines()                 # stripped, non-blank
        if len(raw_lines) < 2:
            raise ValueError("inpcrd too short")

        title = raw_lines[0]
        header_tokens = raw_lines[1].split()
        n_atoms = int(header_tokens[0])
        time = float(header_tokens[1]) if len(header_tokens) > 1 else None

        # ---------- coordinates ----------------------------------------
        coord_vals, line_used = _eat_section(raw_lines[2:], n_atoms * 3)
        coords = coord_vals.reshape(n_atoms, 3)
        cursor = 2 + line_used

        # ---------- velocities (optional) ------------------------------
        velocity_vals = None
        if cursor < len(raw_lines):
            maybe_vels, line_used = _eat_section(raw_lines[cursor:], n_atoms * 3)
            # Heuristic: if the number of lines consumed equals coord section,
            # we assume velocities exist.
            if line_used * 6 >= n_atoms * 3:
                velocity_vals = maybe_vels.reshape(n_atoms, 3)
                cursor += line_used

        # ---------- box (optional) -------------------------------------
        box = mp.Box()
        if cursor < len(raw_lines):
            box_floats = [float(x) for x in raw_lines[cursor].split()]
            if len(box_floats) >= 3:
                box = mp.Box(matrix=np.diag(box_floats[:3]))

        # ---------- populate frame -------------------------------------
        # If a matching atoms block exists, only replace xyz(/vel)
        if "atoms" in frame and len(frame["atoms"]["xyz"]) == n_atoms:
            frame["atoms"]["xyz"] = coords
            if velocity_vals is not None:
                frame["atoms"]["vel"] = velocity_vals
        else:
            atoms_blk = Block(
                {
                    "id": np.arange(1, n_atoms + 1, dtype=int),
                    "name": np.array([f"ATM{i+1}" for i in range(n_atoms)], "U6"),
                    "xyz": coords,
                }
            )
            if velocity_vals is not None:
                atoms_blk["vel"] = velocity_vals
            frame["atoms"] = atoms_blk

        frame.box = box
        if time is not None:
            frame.metadata["timestep"] = int(time)
        frame.metadata["title"] = title

        return frame