from io import StringIO
from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
import nesteddict as nd
import molpy as mp

from .base import TrajectoryReader


class LammpsTrajectoryReader(TrajectoryReader):
    """Reader for LAMMPS trajectory files, supporting multiple files."""

    def __init__(self, fpaths: Union[str, Path, List[Union[str, Path]]]):
        super().__init__(fpaths)
        self._open_all_files()

    def read_frame(self, index: int) -> dict:
        assert index < len(self._byte_offsets), f"Frame index {index} out of range ({len(self._byte_offsets)})"

        mm, offset = self.get_file_and_offset(index)
        mm.seek(offset)

        frame_lines = []
        end_offset = None
        if index + 1 < len(self._byte_offsets):
            next_file, next_offset = self._byte_offsets[index + 1]
            if next_file == self._byte_offsets[index][0]:
                end_offset = next_offset

        for line in iter(mm.readline, b""):
            if end_offset is not None and mm.tell() >= end_offset:
                break
            frame_lines.append(line.decode("utf-8"))

        return self._parse_frame(frame_lines)

    def _parse_trajectories(self):
        self._open_all_files()
        for file_idx, mm in enumerate(self._fp_list):
            mm.seek(0)
            while True:
                pos = mm.tell()
                line = mm.readline()
                if not line:
                    break
                if line.strip().startswith(b"ITEM: TIMESTEP"):
                    self._byte_offsets.append((file_idx, pos))

    def _parse_frame(self, frame_lines: Iterable[str]) -> mp.Frame:
        header = []
        box_bounds = []
        timestep = int(frame_lines[1].strip())

        for i, line in enumerate(frame_lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                periodic = line.split()[-3:]
                for j in range(3):
                    box_bounds.append(list(map(float, frame_lines[i + j + 1].strip().split())))
            elif line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                data_start = i + 1
                break

        df = nd.ArrayDict.from_csv(frame_lines, header=header)
        box_bounds = np.array(box_bounds)

        if box_bounds.shape == (3, 2):
            box_matrix = np.array([
                [box_bounds[0, 1] - box_bounds[0, 0], 0, 0],
                [0, box_bounds[1, 1] - box_bounds[1, 0], 0],
                [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
            ])
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        elif box_bounds.shape == (3, 3):
            xy, xz, yz = box_bounds[:, 2]
            box_matrix = np.array([
                [box_bounds[0, 1] - box_bounds[0, 0], xy, xz],
                [0, box_bounds[1, 1] - box_bounds[1, 0], yz],
                [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
            ])
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        else:
            raise ValueError(f"Invalid box bounds shape {box_bounds.shape}")

        box = mp.Box(matrix=box_matrix, origin=origin)
        return mp.Frame({"atoms": df, "box": box, "timestep": timestep},)
