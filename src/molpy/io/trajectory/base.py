from abc import ABC, abstractmethod
from typing import Iterator, Union, List, Optional, TYPE_CHECKING
from pathlib import Path
import mmap

if TYPE_CHECKING:
    from ...core.trajectory import Trajectory
    from ...core.frame import Frame

PathLike = Union[str, bytes]  # type_check_only

class TrajectoryReader(ABC):
    """
    Base class for trajectory file readers with lazy loading and caching support.
    
    This class provides memory-mapped file reading and works with Trajectory objects
    to enable on-demand frame loading and caching.
    """

    def __init__(self, trajectory: "Trajectory", fpaths: Union[Path, List[Path]]):
        """
        Initialize the trajectory reader.
        
        Args:
            trajectory: Trajectory object to populate with frames
            fpaths: Path or list of paths to trajectory files
        """
        self.trajectory = trajectory
        self.fpaths = [Path(p) for p in (fpaths if isinstance(fpaths, list) else [fpaths])]
        for f in self.fpaths:
            if not f.exists():
                raise FileNotFoundError(f"File not found: {f}")

        self._byte_offsets: List[tuple[int, int]] = []  # list of (file_idx, offset)
        self._fp_list = []  # list of mmap objects
        self._frame_file_index = []  # maps global frame index to file index
        self._total_frames = 0

        self._parse_trajectories()
        self.trajectory.set_total_frames(self._total_frames)

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for mm in self._fp_list:
            if mm is not None:
                mm.close()

    def load_frame(self, index: int) -> "Frame":
        """
        Load a specific frame into the trajectory.
        
        Args:
            index: Frame index to load
            
        Returns:
            The loaded Frame object
        """
        if index < 0:
            index = self._total_frames + index
            
        if index < 0 or index >= self._total_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self._total_frames})")
            
        # Check if frame is already loaded
        if self.trajectory.is_loaded(index):
            # Use int indexing to ensure we get a Frame, not a Trajectory slice
            from ...core.frame import Frame
            frame = self.trajectory.frames[index]  # Direct access to frame dict
            return frame
            
        # Load the frame
        frame = self.read_frame(index)
        self.trajectory._add_frame(index, frame)
        return frame

    def load_frames(self, indices: List[int]) -> List["Frame"]:
        """
        Load multiple frames into the trajectory.
        
        Args:
            indices: List of frame indices to load
            
        Returns:
            List of loaded Frame objects
        """
        return [self.load_frame(i) for i in indices]

    def load_range(self, start: int, stop: int, step: int = 1) -> List["Frame"]:
        """
        Load a range of frames into the trajectory.
        
        Args:
            start: Starting frame index
            stop: Stopping frame index (exclusive)
            step: Step size
            
        Returns:
            List of loaded Frame objects
        """
        indices = list(range(start, stop, step))
        return self.load_frames(indices)

    def preload_all(self) -> None:
        """Load all frames into the trajectory."""
        for i in range(self._total_frames):
            if not self.trajectory.is_loaded(i):
                self.load_frame(i)

    @abstractmethod
    def read_frame(self, index: int) -> "Frame":
        """
        Read a frame from file without adding it to the trajectory.
        
        Args:
            index: Frame index to read
            
        Returns:
            Frame object
        """
        pass

    def read_frames(self, indices: List[int]) -> List["Frame"]:
        """Read multiple frames from file without adding them to the trajectory."""
        return [self.read_frame(i) for i in indices]

    def __len__(self) -> int:
        return self._total_frames

    def __iter__(self) -> Iterator["Frame"]:
        """Iterate over all frames, loading them as needed."""
        for i in range(self._total_frames):
            yield self.load_frame(i)

    @abstractmethod
    def _parse_trajectories(self):
        """Parse multiple trajectory files, storing frame offsets and file index mapping."""
        pass

    def _open_all_files(self):
        """Open all trajectory files with memory mapping."""
        self._fp_list.clear()
        for f in self.fpaths:
            fp = open(f, "r+b")
            # Check file size before mmap
            fp.seek(0, 2)  # Seek to end
            file_size = fp.tell()
            fp.seek(0)  # Seek back to beginning
            
            if file_size == 0:
                # For empty files, store a placeholder and handle in read_frame
                self._fp_list.append(None)
                continue
            
            mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
            self._fp_list.append(mm)

    def get_file_and_offset(self, global_index: int) -> tuple[mmap.mmap, int]:
        """Get memory-mapped file and byte offset for a given frame index."""
        if global_index >= len(self._byte_offsets):
            raise IndexError(f"Frame index {global_index} out of range")
        file_idx, offset = self._byte_offsets[global_index]
        mm = self._fp_list[file_idx]
        if mm is None:
            raise ValueError(f"Empty file at index {file_idx}")
        return mm, offset


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
    def write_frame(self, frame: "Frame"):
        """Write a single frame to the file."""
        pass

    def close(self):
        self._fp.close()

