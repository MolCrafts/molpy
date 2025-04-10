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

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)

    def read_frame(self, index: int) -> dict:
        """Read a specific frame from the trajectory."""
        assert index < len(self._byte_offsets), f"required {index} frame out of range {len(self._byte_offsets)}"
        if self.closed:
            self.read()

        # Seek to the byte offset of the start of the frame
        self._fp.seek(self._byte_offsets[index])
        frame_lines = []

        # Determine the end byte offset
        end_byte_offset = self._byte_offsets[index + 1] if index + 1 < len(self._byte_offsets) else None

        for line in iter(self._fp.readline, b""):
            current_byte_offset = self._fp.tell()
            if end_byte_offset is not None and current_byte_offset >= end_byte_offset:
                break
            frame_lines.append(line.decode("utf-8"))

        return self._parse_frame(frame_lines)
    
    @property
    def closed(self):
        return self._fp.closed

    def _parse_frame(self, frame_lines: Iterable[str]) -> mp.Frame:
        # Initialize variables
        header = []
        data = []
        box_bounds = []
        # Iterate over the lines to parse header, box, timestep, and data
        timestep = int(frame_lines[0].strip())
        for i, line in enumerate(frame_lines):
            if line.startswith("ITEM: BOX BOUNDS"):
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
        """Parse the trajectory file to cache frame start byte offsets."""
        self.read()
        for line in iter(self._fp.readline, b""):
            if self._is_frame_start(line.decode("utf-8")):
                self._byte_offsets.append(self._fp.tell())

    def _is_frame_start(self, line: str) -> bool:
        """Check if a line indicates the start of a frame."""
        return line.strip().startswith("ITEM: TIMESTEP")
