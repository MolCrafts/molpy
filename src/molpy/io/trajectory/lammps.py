from io import StringIO
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
from molpy.core.frame import _dict_to_dataset
import pandas as pd
import molpy as mp

from .base import TrajectoryReader


class LammpsTrajectoryReader(TrajectoryReader):
    """Reader for LAMMPS trajectory files, supporting multiple files."""

    def __init__(self, fpaths: Union[str, Path, List[Union[str, Path]]]):
        super().__init__(fpaths)
        self._open_all_files()

    def read_frame(self, index: int) -> dict:
        assert index < len(
            self._byte_offsets
        ), f"Frame index {index} out of range ({len(self._byte_offsets)})"

        mm, offset = self.get_file_and_offset(index)
        mm.seek(offset)

        file_idx, start_offset = self._byte_offsets[index]

        # 如果有下一个 offset，就用它作为 end
        if (
            index + 1 < len(self._byte_offsets)
            and self._byte_offsets[index + 1][0] == file_idx
        ):
            end_offset = self._byte_offsets[index + 1][1]
        else:
            end_offset = None  # 读到文件尾

        mm = self._fp_list[file_idx]

        if end_offset is None:
            mm.seek(start_offset)
            frame_bytes = mm.read()  # 读到结尾
        else:
            frame_bytes = mm[start_offset:end_offset]  # 一次性读取所需段

        frame_lines = frame_bytes.decode("utf-8").splitlines()

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

    def _parse_frame(self, frame_lines: Sequence[str]) -> mp.Frame:
        header = []
        box_bounds = []
        timestep = int(frame_lines[1].strip())

        for i, line in enumerate(frame_lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                periodic = line.split()[-3:]
                for j in range(3):
                    box_bounds.append(
                        list(map(float, frame_lines[i + j + 1].strip().split()))
                    )
            elif line.startswith("ITEM: ATOMS"):
                header = line.split()[2:]
                data_start = i + 1
                break

        df = pd.read_csv(
            StringIO("\n".join(frame_lines[data_start:])),
            delim_whitespace=True,
            names=header,
        )
        df = _dict_to_dataset({k: df[k].to_numpy() for k in header})
        box_bounds = np.array(box_bounds)

        if box_bounds.shape == (3, 2):
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], 0, 0],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], 0],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        elif box_bounds.shape == (3, 3):
            xy, xz, yz = box_bounds[:, 2]
            box_matrix = np.array(
                [
                    [box_bounds[0, 1] - box_bounds[0, 0], xy, xz],
                    [0, box_bounds[1, 1] - box_bounds[1, 0], yz],
                    [0, 0, box_bounds[2, 1] - box_bounds[2, 0]],
                ]
            )
            origin = np.array([box_bounds[0, 0], box_bounds[1, 0], box_bounds[2, 0]])
        else:
            raise ValueError(f"Invalid box bounds shape {box_bounds.shape}")

        box = mp.Box(matrix=box_matrix, origin=origin)
        return mp.Frame(
            {"atoms": df, "box": box, "timestep": timestep},
        )
