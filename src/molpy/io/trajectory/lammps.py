from pathlib import Path

import numpy as np

from molrs import Frame

from molpy._frame_meta import get_frame_meta

from .base import TrajectoryWriter


class LammpsTrajectoryWriter(TrajectoryWriter):
    """Writer for LAMMPS trajectory files."""

    def __init__(self, fpath: str | Path, atom_style: str = "full"):
        super().__init__(fpath)
        self.atom_style = atom_style

    def write_frame(self, frame: "Frame", timestep: int | None = None):
        """Write a single frame to the file."""
        if timestep is None:
            timestep = get_frame_meta(frame, "timestep", 0)

        # Write timestep
        if self._fp is None:
            raise ValueError("File is not open for writing.")
        self._fp.write(f"ITEM: TIMESTEP\n{timestep}\n".encode())

        # Write number of atoms
        if "atoms" in frame:
            atoms = frame["atoms"]
            first_col = next(iter(atoms.keys()))
            n_atoms = len(atoms[first_col])
            self._fp.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n".encode())

        # Write box bounds
        # The simulation box is a first-class Frame field.
        box = frame.box
        if box:
            matrix = box.matrix
            origin = box.origin

            # Check if box is orthogonal
            if np.allclose(matrix, np.diag(np.diag(matrix))):
                # Orthogonal box
                self._fp.write(b"ITEM: BOX BOUNDS pp pp pp\n")
                for i in range(3):
                    self._fp.write(f"{origin[i]} {origin[i] + matrix[i, i]}\n".encode())
            else:
                # Triclinic box
                self._fp.write(b"ITEM: BOX BOUNDS pp pp pp xy xz yz\n")
                for i in range(3):
                    if i == 0:
                        self._fp.write(
                            f"{origin[i]} {origin[i] + matrix[i, i]} {matrix[0, 1]}\n".encode()
                        )
                    elif i == 1:
                        self._fp.write(
                            f"{origin[i]} {origin[i] + matrix[i, i]} {matrix[0, 2]}\n".encode()
                        )
                    else:
                        self._fp.write(
                            f"{origin[i]} {origin[i] + matrix[i, i]} {matrix[1, 2]}\n".encode()
                        )

        # Write atoms
        if "atoms" in frame:
            atoms = frame["atoms"]

            # Determine column order based on available data
            columns = []
            if "id" in atoms:
                columns.append("id")
            if "mol_id" in atoms:
                columns.append("mol_id")
            if "type" in atoms:
                columns.append("type")
            if "q" in atoms:
                columns.append("q")
            if "x" in atoms and "y" in atoms and "z" in atoms:
                columns.extend(["x", "y", "z"])
            elif "xu" in atoms and "yu" in atoms and "zu" in atoms:
                columns.extend(["xu", "yu", "zu"])
            elif "xs" in atoms and "ys" in atoms and "zs" in atoms:
                columns.extend(["xs", "ys", "zs"])
            if "vx" in atoms and "vy" in atoms and "vz" in atoms:
                columns.extend(["vx", "vy", "vz"])
            if "fx" in atoms and "fy" in atoms and "fz" in atoms:
                columns.extend(["fx", "fy", "fz"])

            # Write atom header
            self._fp.write(f"ITEM: ATOMS {' '.join(columns)}\n".encode())

            # Write atom data
            n_atoms = len(atoms)

            # Get first available column to determine actual atom count
            first_col = next(iter(atoms.keys()))
            actual_n_atoms = len(atoms[first_col])

            for i in range(actual_n_atoms):
                row_data = []
                for col in columns:
                    if col in [
                        "x",
                        "y",
                        "z",
                        "xu",
                        "yu",
                        "zu",
                        "xs",
                        "ys",
                        "zs",
                        "vx",
                        "vy",
                        "vz",
                        "fx",
                        "fy",
                        "fz",
                        "q",
                    ]:
                        row_data.append(f"{atoms[col][i]:.6f}")
                    else:
                        row_data.append(f"{atoms[col][i]}")
                self._fp.write(f"{' '.join(row_data)}\n".encode())

        self._fp.flush()
