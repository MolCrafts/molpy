from pathlib import Path

import numpy as np

from molpy import Block, Box, Frame

from .base import DataReader


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _eat_section(lines: list[str]):
    """
    Consume enough lines to collect *n_values* 12-column floats.
    """
    data = [float(x) for line in lines for x in line.split()]
    return np.asarray(data).reshape(-1, 3)


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
    def read(self, frame: Frame | None = None) -> Frame:
        frame = frame or Frame()

        raw_lines = self.read_lines()  # stripped, non-blank
        if len(raw_lines) < 2:
            raise ValueError("inpcrd too short")

        title = raw_lines[0]
        header_tokens = raw_lines[1].split()
        n_atoms = int(header_tokens[0])
        time = float(header_tokens[1]) if len(header_tokens) > 1 else None

        # ---------- coordinates ----------------------------------------

        cursor = int(n_atoms / 2) + 2
        coords = _eat_section(raw_lines[2:cursor])

        # ---------- velocities (optional) ------------------------------
        velocity_vals = None
        if cursor < len(raw_lines):
            maybe_vels = _eat_section(raw_lines[cursor : cursor + int(n_atoms / 2)])
            # Heuristic: if the number of lines consumed equals coord section,
            # we assume velocities exist.
            velocity_vals = maybe_vels.reshape(n_atoms, 3)
            cursor += int(n_atoms / 2)

        # ---------- box (optional) -------------------------------------
        box = Box()
        if cursor < len(raw_lines):
            box_floats = [float(x) for x in raw_lines[cursor].split()]
            if len(box_floats) >= 3:
                box = Box(matrix=np.diag(box_floats[:3]))

        # ---------- populate frame -------------------------------------
        # If a matching atoms block exists, only replace xyz(/vel)
        if "atoms" in frame:
            assert frame["atoms"].nrows == n_atoms, ValueError(
                f"Frame atoms block has {frame['atoms'].nrows} rows, expected {n_atoms}"
            )
            frame["atoms"]["xyz"] = coords
            if velocity_vals is not None:
                frame["atoms"]["vel"] = velocity_vals
        else:
            atoms_blk = Block(
                {
                    "id": np.arange(1, n_atoms + 1, dtype=int),
                    "name": np.array([f"ATM{i + 1}" for i in range(n_atoms)], "U6"),
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
