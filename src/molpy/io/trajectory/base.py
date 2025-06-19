from abc import ABC, abstractmethod
from typing import Iterator, Union, List
from pathlib import Path
import mmap

import molpy as mp

PathLike = Union[str, bytes]  # type_check_only

class TrajectoryReader(ABC):
    """Base class for trajectory file readers."""

    def __init__(self, fpaths: Union[Path, List[Path]]):
        self.fpaths = [Path(p) for p in (fpaths if isinstance(fpaths, list) else [fpaths])]
        for f in self.fpaths:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")

        self._byte_offsets: List[tuple[int, int]] = []  # list of (file_idx, offset)
        self._fp_list = []  # list of mmap objects
        self._frame_file_index = []  # maps global frame index to file index

        self._parse_trajectories()

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return len(self._byte_offsets)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mm in self._fp_list:
            mm.close()

    @abstractmethod
    def read_frame(self, index: int) -> mp.Frame:
        pass

    def read_frames(self, indices: List[int]) -> List[mp.Frame]:
        return [self.read_frame(i) for i in indices]

    def __len__(self) -> int:
        return len(self._byte_offsets)

    def __iter__(self) -> Iterator[mp.Frame]:
        self._current_frame = 0
        return self

    def __next__(self) -> mp.Frame:
        if self._current_frame >= len(self):
            raise StopIteration
        frame = self.read_frame(self._current_frame)
        self._current_frame += 1
        return frame

    @abstractmethod
    def _parse_trajectories(self):
        """Parse multiple trajectory files, storing frame offsets and file index mapping."""
        pass

    def _open_all_files(self):
        self._fp_list.clear()
        for f in self.fpaths:
            fp = open(f, "r+b")
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            self._fp_list.append(mm)

    def get_file_and_offset(self, global_index: int) -> tuple[mmap.mmap, int]:
        file_idx, offset = self._byte_offsets[global_index]
        return self._fp_list[file_idx], offset

class TrajectoryWriter(ABC):
    """Base class for all chemical file writers."""

    def __init__(self, fpaths: Union[str, Path]):
        self.fpaths = Path(fpaths)
        self._fp = open(self.fpaths, "w+b")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def write_frame(self, frame: mp.Frame):
        """Write a single frame to the file."""
        pass

    def close(self):
        self._fp.close()

