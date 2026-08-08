"""LAMMPS dump trajectory write — molrs-backed.

Incremental :meth:`write_frame` buffers frames; :meth:`close` flushes via
:func:`molrs.io.raw.write_lammps_traj` (requires each frame to carry ``box``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molrs import Frame, MetaValue
import molrs.io.raw as _raw

from molpy._frame_meta import get_frame_meta

from .base import TrajectoryWriter


class LammpsTrajectoryWriter(TrajectoryWriter):
    """Write a LAMMPS dump trajectory (molrs).

    Args:
        fpath: Output path.
        atom_style: Accepted for API parity; molrs derives columns from the frame.
    """

    def __init__(self, fpath: str | Path, atom_style: str = "full") -> None:
        # Base opens a binary handle; we buffer in memory and write once on close.
        super().__init__(fpath)
        self.atom_style = atom_style
        self._frames: list[Frame] = []
        # Do not keep the empty base file open for streaming Python writes.
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def write_frame(self, frame: Frame, timestep: int | None = None) -> None:
        """Buffer one frame (flushed on :meth:`close`).

        Args:
            frame: Frame with coordinates and ``box`` (required by molrs).
            timestep: Optional step index; stored on ``frame.meta['timestep']``.
        """
        if frame.box is None:
            raise ValueError(
                "LAMMPS trajectory write requires frame.box (molrs needs a simbox)"
            )
        if timestep is not None:
            frame.meta = {
                **dict(frame.meta),
                "timestep": MetaValue("i64", int(timestep)),
            }
        elif "timestep" not in frame.meta:
            step = get_frame_meta(frame, "timestep", len(self._frames))
            frame.meta = {
                **dict(frame.meta),
                "timestep": MetaValue("i64", int(step)),
            }
        self._frames.append(frame)

    def close(self) -> None:
        """Flush buffered frames through molrs and release resources."""
        if self._frames:
            _raw.write_lammps_traj(str(self.fpath), self._frames)
            self._frames = []
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
