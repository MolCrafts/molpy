from pathlib import Path

from molpy.core import Block, Frame

from .base import BaseTrajectoryReader, FrameLocation, TrajectoryWriter


class XYZTrajectoryReader(BaseTrajectoryReader):
    """Reader for XYZ trajectory files."""

    def _parse_trajectory(self, file_index: int):
        """Parse XYZ trajectory file to find frame locations."""
        mm = self._mms[file_index]
        fpath = self.fpaths[file_index]

        # Reset to beginning
        mm.seek(0)

        current_offset = 0
        while current_offset < len(mm):
            # Find next frame start (number of atoms line)
            try:
                # Look for a line that contains only a number (atom count)
                line_start = current_offset
                while line_start < len(mm):
                    if mm[line_start] == ord("\n"):
                        line_start += 1
                        break
                    line_start += 1

                if line_start >= len(mm):
                    break

                # Read the atom count line
                line_end = line_start
                while line_end < len(mm) and mm[line_end] != ord("\n"):
                    line_end += 1

                if line_end >= len(mm):
                    break

                atom_count_line = mm[line_start:line_end].decode().strip()
                try:
                    n_atoms = int(atom_count_line)
                except ValueError:
                    # Not a valid atom count, skip
                    current_offset = line_end + 1
                    continue

                # Calculate frame size: atom count + comment + atoms + 2 newlines
                frame_size = len(atom_count_line) + 1  # atom count + newline

                # Skip comment line
                comment_start = line_end + 1
                comment_end = comment_start
                while comment_end < len(mm) and mm[comment_end] != ord("\n"):
                    comment_end += 1
                frame_size += comment_end - comment_start + 1  # comment + newline

                # Skip atom lines
                atom_start = comment_end + 1
                for _ in range(n_atoms):
                    atom_end = atom_start
                    while atom_end < len(mm) and mm[atom_end] != ord("\n"):
                        atom_end += 1
                    frame_size += atom_end - atom_start + 1  # atom line + newline
                    atom_start = atom_end + 1

                # Record frame location
                self._frame_locations.append(
                    FrameLocation(
                        file_index=file_index,
                        byte_offset=current_offset,
                        file_path=fpath,
                    )
                )

                # Move to next potential frame
                current_offset += frame_size

            except Exception:
                # If parsing fails, move forward and try again
                current_offset += 1

        self._total_frames = len(self._frame_locations)

    def _parse_frame(self, frame_lines: list[str]) -> Frame:
        """Parse XYZ frame lines into a Frame object."""
        if not frame_lines:
            return Frame()

        # First line should be atom count
        try:
            n_atoms = int(frame_lines[0])
        except (ValueError, IndexError):
            raise ValueError("Invalid XYZ frame: first line must be atom count")

        # Second line is comment (optional)
        comment_line = frame_lines[1] if len(frame_lines) > 1 else ""

        # Remaining lines are atoms
        if len(frame_lines) < 2 + n_atoms:
            raise ValueError(
                f"Invalid XYZ frame: expected {n_atoms} atoms, got {len(frame_lines) - 2}"
            )

        # Create frame
        frame = Frame()

        # Parse atoms
        atoms_data = []
        for i in range(n_atoms):
            atom_line = frame_lines[2 + i]
            parts = atom_line.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid atom line {i + 1}: {atom_line}")

            try:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])

                atom = {"element": element, "x": x, "y": y, "z": z, "id": i + 1}
                atoms_data.append(atom)
            except ValueError:
                raise ValueError(
                    f"Invalid coordinates in atom line {i + 1}: {atom_line}"
                )

        # Add atoms to frame
        if atoms_data:
            import numpy as np

            # Extract arrays
            elements = [atom["element"] for atom in atoms_data]
            x_coords = np.array([atom["x"] for atom in atoms_data])
            y_coords = np.array([atom["y"] for atom in atoms_data])
            z_coords = np.array([atom["z"] for atom in atoms_data])
            ids = np.array([atom["id"] for atom in atoms_data])

            # Create block
            Block(
                {
                    "element": elements,
                    "x": x_coords,
                    "y": y_coords,
                    "z": z_coords,
                    "id": ids,
                },
            )

        # Store comment if present
        if comment_line:
            frame.metadata.setdefault("comment", comment_line)

        return frame


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
        box = frame.metadata.get("box", None)
        n_atoms = len(atoms)

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
