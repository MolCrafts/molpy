from abc import abstractmethod
from collections.abc import Iterable, Iterator
from pathlib import Path

from molpy.core import Frame

from ..base import BaseReader, PathLike

__all__ = [
    "BaseTrajectoryReader",
    "TrajectoryWriter",
    "PathLike",
]


class BaseTrajectoryReader(BaseReader, Iterable["Frame"]):
    """Pure, storage-agnostic trajectory reader: a lazy ``Iterable[Frame]``.

    Subclasses implement only ``read_frame(index)`` and the ``n_frames``
    property; the random-access and iteration API (``__iter__``,
    ``__getitem__``, slicing, ``read_frames`` / ``read_range`` / ``read_all``,
    ``__len__``) is derived entirely from those two and involves no files.
    """

    @property
    @abstractmethod
    def n_frames(self) -> int:
        """Total number of frames in the trajectory."""

    @abstractmethod
    def read_frame(self, index: int) -> "Frame":
        """Read and return the frame at ``index`` (negative indices allowed)."""

    def read_frames(self, indices: list[int]) -> list["Frame"]:
        """Read the frames at ``indices``.

        Args:
            indices: Frame indices to read.

        Returns:
            The frames, in the order requested.
        """
        return [self.read_frame(i) for i in indices]

    def read_range(self, start: int, stop: int, step: int = 1) -> list["Frame"]:
        """Read frames ``start`` (inclusive) to ``stop`` (exclusive) by ``step``."""
        return self.read_frames(list(range(start, stop, step)))

    def read_all(self) -> list["Frame"]:
        """Read every frame in the trajectory."""
        return [self.read_frame(i) for i in range(self.n_frames)]

    def __len__(self) -> int:
        return self.n_frames

    def __iter__(self) -> Iterator["Frame"]:
        """Iterate over all frames lazily."""
        for i in range(self.n_frames):
            yield self.read_frame(i)

    def __getitem__(self, index: int | slice) -> "Frame | list[Frame]":
        """Support integer indexing and slicing of frames."""
        if isinstance(index, int):
            return self.read_frame(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.n_frames)
            return self.read_range(start, stop, step)
        else:
            raise TypeError("Index must be int or slice")


class TrajectoryWriter(BaseReader):
    """Base class for all trajectory file writers."""

    def __init__(self, fpath: str | Path):
        super().__init__(fpath, must_exist=False)
        self._fp = open(self.fpath, "w+b")

    @abstractmethod
    def write_frame(self, frame: "Frame"):
        """Write a single frame to the file."""
        pass

    def close(self):
        if hasattr(self, "_fp") and self._fp is not None:
            self._fp.close()
            self._fp = None
