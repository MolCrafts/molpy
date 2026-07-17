from pathlib import Path
from io import TextIOWrapper

from molrs import Frame

from molpy._frame_meta import get_frame_meta

from .base import TrajectoryWriter


class XYZTrajectoryWriter(TrajectoryWriter):
    """Writer for XYZ trajectory files."""

    def __init__(self, fpath: str | Path):
        super().__init__(fpath)
        # Reuse TrajectoryWriter's owned binary handle instead of opening a
        # second descriptor that its base lifecycle cannot close.
        self.fobj = TextIOWrapper(self._fp, encoding="utf-8", write_through=True)

    def __del__(self):
        if hasattr(self, "fobj") and not self.fobj.closed:
            self.fobj.close()

    def write_frame(self, frame: Frame):
        """Write a single frame to the XYZ file."""
        atoms = frame["atoms"]
        box = frame.simbox
        n_atoms = atoms.nrows

        self.fobj.write(f"{n_atoms}\n")

        # Write comment line
        comment = get_frame_meta(
            frame, "comment", f"Step={get_frame_meta(frame, 'step', 0)}"
        )
        if box is not None:
            comment += f' Lattice="{box.matrix.tolist()}"'
        self.fobj.write(f"{comment}\n")

        for _, atom in atoms.iterrows():
            x = atom["x"]
            y = atom["y"]
            z = atom["z"]
            elem = atom.get("element", "X")

            self.fobj.write(f"{elem} {x} {y} {z}\n")

    def write_traj(self, trajectory):
        """Write multiple frames to the XYZ file."""
        for frame in trajectory:
            self.write_frame(frame)

    def close(self):
        """Close the file."""
        if hasattr(self, "fobj") and not self.fobj.closed:
            self.fobj.close()
        self._fp = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
