from pathlib import Path

import numpy as np

from molpy.core.box import Box
from molpy.core.frame import Frame, Block

from .base import DataReader


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _parse_fixed_width(raw: str, width: int = 12) -> list[float]:
    """Parse a line using fixed-width columns. Raises ValueError on failure."""
    tokens = [raw[i : i + width].strip() for i in range(0, len(raw), width)]
    return [float(t) for t in tokens if t]


def _eat_section(lines: list[str]):
    """
    Consume enough lines to collect floats from AMBER coordinate format.

    AMBER inpcrd uses Fortran ``6F12.7`` — 12 characters per value, 6 values
    per line.  Whitespace splitting fails when a negative sign abuts the
    preceding value (e.g. ``50.5413286-100.7101036``).  We try fixed-width
    parsing first and fall back to whitespace splitting if that fails.
    """
    data: list[float] = []
    for line in lines:
        raw = line.rstrip("\n")
        if not raw.strip():
            continue
        try:
            data.extend(_parse_fixed_width(raw))
        except ValueError:
            data.extend(float(x) for x in raw.split() if x)
    return np.asarray(data).reshape(-1, 3)


# ---------------------------------------------------------------------
# main reader
# ---------------------------------------------------------------------
class AmberInpcrdReader(DataReader):
    """
    Reader for AMBER ASCII `*.inpcrd` (old-style) coordinate files.

    * Coordinates: 12.7/12.8 format, 6 numbers per line
    * Optional velocities section (same length as coordinates)
    * Optional final box line (3-6 floats)
    """

    __slots__ = ()

    def __init__(self, file: str | Path, **kwargs):
        super().__init__(path=Path(file), **kwargs)

    # ------------------------------------------------------------------
    def read(self, frame: Frame | None = None) -> Frame:
        frame = frame or Frame()

        raw_lines = self.read_lines()
        if len(raw_lines) < 2:
            raise ValueError("inpcrd too short")

        title = raw_lines[0].strip()
        header_tokens = raw_lines[1].strip().split()
        n_atoms = int(header_tokens[0])
        time = float(header_tokens[1]) if len(header_tokens) > 1 else None

        # ---------- coordinates ----------------------------------------
        # AMBER inpcrd format: 6 values per line, each 12 chars wide
        # Number of coordinate lines = ceil(n_atoms * 3 / 6)
        n_coord_lines = (n_atoms * 3 + 5) // 6

        if len(raw_lines) < 2 + n_coord_lines:
            raise ValueError(
                f"Not enough lines for {n_atoms} atoms: "
                f"need {n_coord_lines} coordinate lines, "
                f"got {len(raw_lines) - 2}"
            )

        cursor = 2 + n_coord_lines
        coords = _eat_section(raw_lines[2:cursor])

        # ---------- velocities (optional) ------------------------------
        # Per AMBER spec, velocities only appear in restart files (those
        # with a timestamp on the header line).
        velocity_vals = None
        if time is not None:
            non_blank_remaining = sum(1 for line in raw_lines[cursor:] if line.strip())
            if non_blank_remaining >= n_coord_lines:
                maybe_vels = _eat_section(raw_lines[cursor : cursor + n_coord_lines])
                if maybe_vels.size == n_atoms * 3:
                    velocity_vals = maybe_vels.reshape(n_atoms, 3)
                    cursor += n_coord_lines

        # ---------- box (optional) -------------------------------------
        box = Box()
        if cursor < len(raw_lines):
            raw = raw_lines[cursor].rstrip("\n")
            try:
                box_floats = _parse_fixed_width(raw)
            except ValueError:
                box_floats = [float(x) for x in raw.split() if x]
            if len(box_floats) >= 3:
                box = Box(matrix=np.diag(box_floats[:3]))

        # ---------- populate frame -------------------------------------
        # If a matching atoms block exists, only replace coordinates
        if "atoms" in frame:
            if frame["atoms"].nrows != n_atoms:
                raise ValueError(
                    f"atoms block has {frame['atoms'].nrows} rows, but inpcrd has {n_atoms}"
                )
            # Store coordinates as separate x, y, z fields
            frame["atoms"]["x"] = coords[:, 0]
            frame["atoms"]["y"] = coords[:, 1]
            frame["atoms"]["z"] = coords[:, 2]
            if velocity_vals is not None:
                frame["atoms"]["vel"] = velocity_vals
        else:
            atoms_blk = Block(
                {
                    "id": np.arange(1, n_atoms + 1, dtype=int),
                    "name": np.array([f"ATM{i + 1}" for i in range(n_atoms)], "U6"),
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "z": coords[:, 2],
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
