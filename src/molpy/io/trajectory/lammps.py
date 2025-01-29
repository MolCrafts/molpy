from .base import TrajectoryReader
from pathlib import Path
import mmap
import molpy as mp
import pandas as pd
import numpy as np
from typing import Iterable
from io import StringIO

class LammpsTrajectoryReader(TrajectoryReader):
    """Reader for LAMMPS trajectory files."""

    def __init__(self, filepath: str | Path, cache_size: int | None = None):
        super().__init__(filepath, cache_size)

    def read_frame(self, index: int) -> dict:
        """Read a specific frame from the trajectory."""
        start_line = self._frames_start[index]
        end_line = self._frames_end[index]

        with open(self.filepath, "r+b") as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            mmapped_file.seek(0)  # Ensure we start from the beginning of the file
            current_line = 0
            frame_lines = []

            for line in iter(mmapped_file.readline, b""):
                if start_line <= current_line <= end_line:
                    frame_lines.append(line.decode("utf-8"))
                elif current_line > end_line:
                    break
                current_line += 1

            mmapped_file.close()

        return self._parse_frame(frame_lines)

    def _parse_frame(self, frame_lines: Iterable[str]) -> mp.Frame:
        # Initialize variables
        header = []
        data = []
        box_bounds = []
        timestep = None

        # Iterate over the lines to parse header, box, timestep, and data
        for i, line in enumerate(frame_lines):
            if line.startswith("ITEM: TIMESTEP"):
                timestep = int(frame_lines[i+1].strip())
            elif line.startswith("ITEM: BOX BOUNDS"):
                box_line = line.split()
                # box_header = box_line[3:-3]
                # if "abc" in box_header and "origin" in box_header:
                #     box_type = "general"
                # elif ""
                periodic = box_line[-3:]
                for j in range(3):
                    box_bounds.append(
                        list(map(float, frame_lines[i+j+1].strip().split()))
                    )
            elif line.startswith("ITEM: ATOMS"):
                # Extract column names from the header
                header = line.split()[2:]
                break

        # Use pandas to read the remaining lines as a DataFrame
        data = pd.read_csv(
            StringIO('\n'.join(frame_lines[i+1:])),
            sep=r'\s+',
            names=header,
        )

        # Create the box using mp.Box
        box_bounds = np.array(box_bounds)

        if box_bounds.shape == (3, 2):
            # xlo xhi
            # ylo yhi
            # zlo zhi
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], 0, 0],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], 0],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        elif box_bounds.shape == (3, 3):
            # xlo xhi xy
            # ylo yhi xz
            # zlo zhi yz
            xy = box_bounds[0, 2]
            xz = box_bounds[1, 2]
            yz = box_bounds[2, 2]
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], xy, xz],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], yz],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        else:
            raise ValueError(f"Invalid box bounds shape {box_bounds.shape} {box_bounds}")

        box = mp.Box(matrix=box_matrix, origin=origin)
        # Create and return an mp.Frame
        return mp.Frame({"atoms": data}, box=box, timestep=timestep)

    def _parse_trajectory(self):
        """Parse the trajectory file to cache frame start and end lines."""
        with open(self.filepath, "r+b") as file:
            mmapped_file = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
            line_number = 0
            for line in iter(mmapped_file.readline, b""):
                if self._is_frame_start(line.decode("utf-8")):
                    self._frames_start.append(line_number)
                    if len(self._frames_start) > 1:
                        self._frames_end.append(line_number - 1)
                line_number += 1
            # Add the last frame's end line
            if self._frames_start:
                self._frames_end.append(line_number - 1)
            mmapped_file.close()

    def _is_frame_start(self, line: str) -> bool:
        """Check if a line indicates the start of a frame."""
        return line.strip().startswith("ITEM: TIMESTEP")
