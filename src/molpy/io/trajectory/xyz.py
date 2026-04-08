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
            # Skip any leading whitespace/newlines
            while current_offset < len(mm) and mm[current_offset] in (
                ord("\n"),
                ord("\r"),
                ord(" "),
                ord("\t"),
            ):
                current_offset += 1

            if current_offset >= len(mm):
                break

            try:
                # Read the atom count line (first line of frame)
                line_end = current_offset
                while line_end < len(mm) and mm[line_end] != ord("\n"):
                    line_end += 1

                atom_count_line = mm[current_offset:line_end].decode().strip()
                if not atom_count_line:
                    current_offset = line_end + 1
                    continue

                try:
                    n_atoms = int(atom_count_line)
                except ValueError:
                    # Not a valid atom count, skip this line
                    current_offset = line_end + 1
                    continue

                # Record frame location at the atom count line
                frame_start = current_offset

                # Skip past atom count line
                pos = line_end + 1

                # Skip comment line
                while pos < len(mm) and mm[pos] != ord("\n"):
                    pos += 1
                pos += 1  # skip newline

                # Skip atom lines
                for _ in range(n_atoms):
                    while pos < len(mm) and mm[pos] != ord("\n"):
                        pos += 1
                    pos += 1  # skip newline

                # Record frame location
                self._frame_locations.append(
                    FrameLocation(
                        file_index=file_index,
                        byte_offset=frame_start,
                        file_path=fpath,
                    )
                )

                # Move to next potential frame
                current_offset = pos

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

            # Create block and assign to frame
            frame["atoms"] = Block(
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
        box = frame.box
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
