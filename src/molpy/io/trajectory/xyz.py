from pathlib import Path

from molpy.core import Frame

from .base import TrajectoryWriter


class XYZTrajectoryWriter(TrajectoryWriter):
    """Writer for XYZ trajectory files."""

    def __init__(self, fpath: str | Path):
        super().__init__(fpath)
        self.fobj = open(fpath, "w")

    def __del__(self):
        if hasattr(self, "fobj") and not self.fobj.closed:
            self.fobj.close()

    def write_frame(self, frame: Frame):
        """Write a single frame to the XYZ file."""
        atoms = frame["atoms"]
        box = frame.box
        n_atoms = atoms.nrows

        self.fobj.write(f"{n_atoms}\n")

        # Write comment line
        comment = frame.metadata.get("comment", f"Step={frame.metadata.get('step', 0)}")
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
