from pathlib import Path

import numpy as np

from molpy.core.frame import Block
from molpy.core.box import Box
from molpy.core.frame import Frame
from molpy.core.element import Element

from .base import DataReader, DataWriter


class GroReader(DataReader):
    """
    Robust GROMACS GRO file reader.

    Features:
    - Parses GRO format atoms with proper field extraction
    - Handles box information (orthogonal and triclinic)
    - Robust error handling for malformed files
    - Proper coordinate precision preservation
    """

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        """Clean line while preserving GRO format."""
        return line.rstrip()

    def _parse_atom_line(self, line: str) -> dict | None:
        """
        Parse a single atom line from GRO format.

        GRO format (fixed width):
        positions 1-5: residue number
        positions 6-10: residue name
        positions 11-15: atom name
        positions 16-20: atom number
        positions 21-28: x coordinate (nm)
        positions 29-36: y coordinate (nm)
        positions 37-44: z coordinate (nm)
        positions 45-52: x velocity (optional)
        positions 53-60: y velocity (optional)
        positions 61-68: z velocity (optional)
        """
        if len(line) < 44:
            return None

        try:
            atom_data = {
                "res_number": line[:5].strip(),
                "res_name": line[5:10].strip(),
                "name": line[10:15].strip(),
                "number": int(line[15:20].strip()) if line[15:20].strip() else 0,
                "x": float(line[20:28].strip()) if line[20:28].strip() else 0.0,
                "y": float(line[28:36].strip()) if line[28:36].strip() else 0.0,
                "z": float(line[36:44].strip()) if line[36:44].strip() else 0.0,
            }

            # Handle optional velocity
            if len(line) >= 68:
                try:
                    atom_data["vx"] = (
                        float(line[44:52].strip()) if line[44:52].strip() else 0.0
                    )
                    atom_data["vy"] = (
                        float(line[52:60].strip()) if line[52:60].strip() else 0.0
                    )
                    atom_data["vz"] = (
                        float(line[60:68].strip()) if line[60:68].strip() else 0.0
                    )
                except ValueError:
                    pass  # No velocity data

            return atom_data

        except (ValueError, IndexError):
            return None

    def _parse_box_line(self, line: str) -> np.ndarray:
        """Parse box vectors from the last line."""
        try:
            parts = line.strip().split()
            if len(parts) == 3:
                # Orthogonal box: v1(x) v2(y) v3(z)
                v1x, v2y, v3z = map(float, parts)
                matrix = np.diag([v1x, v2y, v3z])
            elif len(parts) == 9:
                # Triclinic box: v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
                v1x, v2y, v3z, v1y, v1z, v2x, v2z, v3x, v3y = map(float, parts)
                matrix = np.array([[v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]])
            else:
                # Default box
                matrix = np.eye(3)

            return matrix

        except (ValueError, IndexError):
            return np.eye(3)

    def assign_numbers(self, atoms):
        """Assign atomic numbers based on atom names."""
        for atom in atoms:
            if "number" not in atom or atom["number"] == 0:
                # Guess from atom name
                name = atom.get("name", "").strip()
                if name:
                    # Remove numbers from name
                    clean_name = "".join(c for c in name if c.isalpha())
                    if clean_name:
                        try:
                            # Try two-letter element first
                            if len(clean_name) >= 2:
                                try:
                                    element = Element(clean_name[:2].upper())
                                    atom["number"] = element.number
                                    continue
                                except (KeyError, ValueError):
                                    pass
                            # Then try single letter
                            if len(clean_name) >= 1:
                                try:
                                    element = Element(clean_name[:1].upper())
                                    atom["number"] = element.number
                                    continue
                                except (KeyError, ValueError):
                                    pass
                            # Default to hydrogen if nothing works
                            atom["number"] = 1
                        except Exception:
                            atom["number"] = 1  # Default to hydrogen

    def read(self, frame: Frame | None = None) -> Frame:
        """Read GRO file and populate frame."""

        # Read file content
        with open(self._file) as f:
            lines = f.readlines()

        lines = list(map(self.sanitizer, lines))

        # Parse title (first line)
        lines[0] if lines else "Unknown"

        # Parse number of atoms (second line)
        try:
            natoms = int(lines[1]) if len(lines) > 1 else 0
        except (ValueError, IndexError):
            natoms = 0

        # Parse atoms
        atoms_data: dict = {
            "res_number": [],
            "res_name": [],
            "name": [],
            "number": [],
            "xyz": [],
            "vx": [],  # Separate velocity components
            "vy": [],
            "vz": [],
        }

        has_velocity = False

        atom_lines = lines[2 : 2 + natoms] if len(lines) > 2 else []
        for line in atom_lines:
            atom_data = self._parse_atom_line(line)
            if atom_data:
                atoms_data["res_number"].append(atom_data["res_number"])
                atoms_data["res_name"].append(atom_data["res_name"])
                atoms_data["name"].append(atom_data["name"])
                atoms_data["number"].append(atom_data["number"])
                atoms_data["xyz"].append(
                    [atom_data["x"], atom_data["y"], atom_data["z"]]
                )

                if "vx" in atom_data:
                    atoms_data["vx"].append(atom_data["vx"])
                    atoms_data["vy"].append(atom_data["vy"])
                    atoms_data["vz"].append(atom_data["vz"])
                    has_velocity = True
                else:
                    atoms_data["vx"].append(0.0)
                    atoms_data["vy"].append(0.0)
                    atoms_data["vz"].append(0.0)

        # Remove velocity components if not present in any atom
        if not has_velocity:
            del atoms_data["vx"]
            del atoms_data["vy"]
            del atoms_data["vz"]

        # Assign atomic numbers if missing
        if atoms_data["res_number"]:  # Only if we have atoms
            atom_dicts = []
            for i in range(len(atoms_data["res_number"])):
                atom_dict = {
                    key: values[i] if values else None
                    for key, values in atoms_data.items()
                }
                atom_dicts.append(atom_dict)

            self.assign_numbers(atom_dicts)

            # Update atomic numbers
            for i, atom_dict in enumerate(atom_dicts):
                atoms_data["number"][i] = atom_dict["number"]

        # Convert xyz to separate x, y, z fields (keep both formats)
        if "xyz" in atoms_data and atoms_data["xyz"]:
            xyz_array = np.array(atoms_data["xyz"], dtype=float)
            atoms_data["x"] = xyz_array[:, 0]
            atoms_data["y"] = xyz_array[:, 1]
            atoms_data["z"] = xyz_array[:, 2]
            # Keep xyz field for backward compatibility
            atoms_data["xyz"] = xyz_array

        # Convert to numpy arrays
        for key in list(atoms_data.keys()):
            values = atoms_data[key]
            # Check if values is not empty (works for lists and arrays)
            if values is not None and len(values) > 0:
                if key in ["xyz", "velocity"]:
                    # Already numpy array or should be 2D array
                    if not isinstance(values, np.ndarray):
                        atoms_data[key] = np.array(values, dtype=float)
                elif key == "number":
                    atoms_data[key] = np.array(values, dtype=int)
                elif key in ["x", "y", "z", "vx", "vy", "vz"]:
                    # Coordinate/velocity components
                    atoms_data[key] = np.array(values, dtype=float)
                else:
                    # For string data, use proper string dtype
                    max_len = max(len(str(v)) for v in values)
                    atoms_data[key] = np.array(values, dtype=f"U{max_len}")

        # Create dataset
        frame["atoms"] = Block(atoms_data)

        # Parse box from last line
        if len(lines) > 2 + natoms:
            box_line = lines[2 + natoms]
            box_matrix = self._parse_box_line(box_line)
            frame.box = Box(matrix=box_matrix)
        else:
            frame.box = Box()  # Default box

        return frame


class GroWriter(DataWriter):
    """
    GROMACS GRO file writer.

    Features:
    - Writes properly formatted GRO files
    - Handles orthogonal and triclinic boxes
    - Supports velocity information
    - Maintains precision for coordinates
    """

    def __init__(self, path: str | Path):
        super().__init__(Path(path))
        self._path = Path(path)

    def _format_atom_line(
        self,
        res_num: int,
        res_name: str,
        atom_name: str,
        atom_num: int,
        x: float,
        y: float,
        z: float,
        vx: float | None = None,
        vy: float | None = None,
        vz: float | None = None,
    ) -> str:
        """Format a single atom line in GRO format."""

        # Ensure proper field widths
        res_name = res_name[:5].ljust(5)[:5]
        atom_name = atom_name[:5].ljust(5)[:5]

        # Format coordinates (nm, 3 decimal places)
        coord_str = f"{x:8.3f}{y:8.3f}{z:8.3f}"

        # Basic line without velocity
        line = f"{res_num:>5d}{res_name}{atom_name}{atom_num:>5d}{coord_str}"

        # Add velocity if provided
        if vx is not None and vy is not None and vz is not None:
            vel_str = f"{vx:8.4f}{vy:8.4f}{vz:8.4f}"
            line += vel_str

        return line

    def _format_box_line(self, box) -> str:
        """Format box line from box object."""
        if box is None:
            return "   1.00000   1.00000   1.00000"

        matrix = box.matrix

        # Check if orthogonal
        if np.allclose(matrix, np.diag(np.diag(matrix))):
            # Orthogonal box
            v1x, v2y, v3z = matrix[0, 0], matrix[1, 1], matrix[2, 2]
            return f"{v1x:10.5f}{v2y:10.5f}{v3z:10.5f}"
        else:
            # Triclinic box: v1(x) v2(y) v3(z) v1(y) v1(z) v2(x) v2(z) v3(x) v3(y)
            v1x, v1y, v1z = matrix[0, :]
            v2x, v2y, v2z = matrix[1, :]
            v3x, v3y, v3z = matrix[2, :]

            return (
                f"{v1x:10.5f}{v2y:10.5f}{v3z:10.5f}"
                f"{v1y:10.5f}{v1z:10.5f}{v2x:10.5f}"
                f"{v2z:10.5f}{v3x:10.5f}{v3y:10.5f}"
            )

    def _get_atom_data_at_index(self, atoms_dataset, index: int) -> dict:
        """Extract atom data at given index from dataset."""
        atom_data = {}

        # Extract data for this atom
        for var_name in atoms_dataset.data_vars:
            values = atoms_dataset[var_name]

            if var_name == "xyz":
                # Handle coordinate array - find xyz dimension
                xyz_dims = [
                    d
                    for d in values.dims
                    if str(d).startswith("dim_xyz_") and not str(d).endswith("_1")
                ]
                if xyz_dims:
                    xyz_dim = xyz_dims[0]
                    coord = values.isel({xyz_dim: index}).values
                    atom_data["x"] = float(coord[0])
                    atom_data["y"] = float(coord[1])
                    atom_data["z"] = float(coord[2])
            elif var_name == "velocity":
                # Handle velocity array - find velocity dimension
                vel_dims = [
                    d
                    for d in values.dims
                    if str(d).startswith("dim_velocity_") and not str(d).endswith("_1")
                ]
                if vel_dims:
                    vel_dim = vel_dims[0]
                    vel = values.isel({vel_dim: index}).values
                    atom_data["vx"] = float(vel[0])
                    atom_data["vy"] = float(vel[1])
                    atom_data["vz"] = float(vel[2])
            else:
                # Handle scalar values - find the specific dimension for this variable
                var_dims = [
                    d for d in values.dims if str(d).startswith(f"dim_{var_name}_")
                ]
                if var_dims:
                    var_dim = var_dims[0]
                    value = values.isel({var_dim: index}).values
                    if hasattr(value, "item"):
                        value = value.item()
                    atom_data[var_name] = value

        return atom_data

    def write(self, frame):
        """Write frame to GRO file."""

        with open(self._path, "w") as f:
            # Write title line
            title = frame.metadata.get("name", "Generated by molpy")
            if hasattr(frame, "timestep") and frame.timestep is not None:
                title += f", t= {frame.timestep}"
            f.write(title + "\n")

            # Count atoms
            n_atoms = 0
            if "atoms" in frame:
                atoms = frame["atoms"]
                n_atoms = atoms.nrows

            # Write number of atoms
            f.write(f"{n_atoms:>5d}\n")

            # Write atoms
            if n_atoms > 0:
                atoms = frame["atoms"]
                has_velocity = "vx" in atoms and "vy" in atoms and "vz" in atoms

                for i in range(n_atoms):
                    atom_data = atoms[i]  # Get atom as dict

                    # Extract required fields with defaults
                    res_num = atom_data.get("res_number", 1)
                    if isinstance(res_num, str):
                        try:
                            res_num = int(res_num)
                        except ValueError:
                            res_num = 1
                    else:
                        res_num = int(res_num)

                    res_name = str(atom_data.get("res_name", "MOL"))
                    atom_name = str(atom_data.get("name", "X"))
                    atom_num = int(atom_data.get("number", i + 1))

                    # Handle coordinates - support both xyz array and separate x, y, z fields
                    if "x" in atom_data and "y" in atom_data and "z" in atom_data:
                        x = float(atom_data["x"])
                        y = float(atom_data["y"])
                        z = float(atom_data["z"])
                    elif "xyz" in atom_data:
                        xyz = atom_data["xyz"]
                        if hasattr(xyz, "__iter__") and not isinstance(xyz, str):
                            x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                        else:
                            raise ValueError(f"Invalid xyz format for atom {i}")
                    else:
                        raise ValueError(
                            f"Atom {i} missing coordinate information (need x/y/z or xyz)"
                        )

                    # Velocity (optional)
                    vx = float(atom_data.get("vx", 0.0)) if has_velocity else None
                    vy = float(atom_data.get("vy", 0.0)) if has_velocity else None
                    vz = float(atom_data.get("vz", 0.0)) if has_velocity else None

                    line = self._format_atom_line(
                        res_num, res_name, atom_name, atom_num, x, y, z, vx, vy, vz
                    )
                    f.write(line + "\n")

            # Write box line
            box_line = self._format_box_line(getattr(frame, "box", None))
            f.write(box_line + "\n")
