from abc import ABC, abstractmethod
from typing import Iterator, Union, List
from pathlib import Path
import mmap

class BaseReader(ABC):
    """Base class for all chemical file readers."""

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ...


class TrajectoryReader(BaseReader):
    """Base class for trajectory file readers."""

    def __init__(self, filepath: str | Path):
        super().__init__(filepath)
        self._byte_offsets: list[int] = []
        self._parse_trajectory()

    @abstractmethod
    def read_frame(self, index: int) -> dict:
        """
        Read a specific frame from the trajectory.

        Args:
            index: Frame index (0-based)

        Returns:
            dict: Frame data containing at minimum:
                - positions: np.ndarray of shape (n_atoms, 3)
                - cell: np.ndarray of shape (3, 3) or None
                - symbols: List[str] of atomic symbols
        """
        pass

    def read_frames(self, indices: List[int]) -> List[dict]:
        """Read multiple frames by their indices."""
        return [self.read_frame(i) for i in indices]
    
    @property
    def n_frames(self):
        return len(self._byte_offsets)

    def __iter__(self) -> Iterator[dict]:
        """Iterate over all frames in the trajectory."""
        self._current_frame = 0  # Initialize the current frame index
        return self

    def __next__(self) -> dict:
        """Get the next frame in iteration."""
        if self._current_frame >= self.n_frames:  # Use the length of frames_start for iteration
            raise StopIteration

        frame = self.read_frame(
            self._current_frame
        )  # Read frame using start indices
        self._current_frame += 1
        return frame

    def __len__(self) -> int:
        """Return the number of frames in the trajectory."""
        return len(self._byte_offsets)  # Return the length of frames_start
    
    @abstractmethod
    def _parse_trajectory(self):
        """Parse the trajectory file to cache frame start and end lines."""
        pass

    def read(self):
        fp = open(self.filepath, "r+b")
        mmapped_file = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
        mmapped_file.seek(0)
        self._fp = mmapped_file