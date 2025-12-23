import json
import mmap
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from logging import getLogger as get_logger
from pathlib import Path
from typing import NamedTuple

from molpy.core import Frame

logger = get_logger(__name__)


PathLike = str | Path


class FrameLocation(NamedTuple):
    """Location information for a frame."""

    file_index: int
    byte_offset: int
    file_path: Path


class BaseTrajectoryReader(ABC, Iterable["Frame"]):
    """
    Base class for trajectory file readers that act as lazy-loading iterators.

    This class provides memory-mapped file reading and directly returns Frame objects
    without loading everything into memory. Supports reading from multiple files.

    Implements Iterable[Frame] for lazy iteration over frames.
    """

    def __init__(self, fpath: PathLike | list[PathLike]):
        """
        Initialize the trajectory reader.

        Args:
            fpath: Path to trajectory file or list of paths to multiple trajectory files
        """
        # Handle both single file and multiple files
        if isinstance(fpath, (str, Path)):
            self.fpaths = [Path(fpath)]
        else:
            self.fpaths = [Path(p) for p in fpath]

        # Validate all files exist
        for path in self.fpaths:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

        self._frame_locations: list[FrameLocation] = []  # location info for each frame
        self._mms: list[mmap.mmap] = []  # memory-mapped file objects for each file
        self._total_frames = 0
        self._index_files = self._get_index_file_paths()
        self._open_files()

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return self._total_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Close all memory-mapped files."""
        for mm in self._mms:
            if mm is not None:
                mm.close()
        self._mms.clear()

    def read_frame(self, index: int) -> "Frame":
        """
        Read a specific frame from the trajectory file(s).

        Args:
            index: Global frame index to read

        Returns:
            The Frame object
        """
        if index < 0:
            index = self._total_frames + index

        if index < 0 or index >= self._total_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self._total_frames})"
            )

        # Get location info for this frame
        location = self._get_frame_location(index)

        # Calculate frame end position
        if index + 1 < len(self._frame_locations):
            next_location = self._frame_locations[index + 1]
            if next_location.file_index == location.file_index:
                frame_end = next_location.byte_offset
            else:
                frame_end = None  # End of file
        else:
            frame_end = None  # Last frame

        # Get the memory-mapped file and read frame data
        mm = self._get_mmap(location.file_index)
        frame_bytes = mm[location.byte_offset : frame_end]
        frame_lines = frame_bytes.decode().splitlines()

        # Parse the frame lines using the derived class implementation
        return self._parse_frame(frame_lines)

    def read_frames(self, indices: list[int]) -> list["Frame"]:
        """
        Read multiple frames from the trajectory file.

        Args:
            indices: list of frame indices to read

        Returns:
            list of Frame objects
        """
        return [self.read_frame(i) for i in indices]

    def read_range(self, start: int, stop: int, step: int = 1) -> list["Frame"]:
        """
        Read a range of frames from the trajectory file.

        Args:
            start: Starting frame index
            stop: Stopping frame index (exclusive)
            step: Step size

        Returns:
            list of Frame objects
        """
        indices = list(range(start, stop, step))
        return self.read_frames(indices)

    def read_all(self) -> list["Frame"]:
        """Read all frames from the trajectory file."""
        return [self.read_frame(i) for i in range(self._total_frames)]

    @abstractmethod
    def _parse_frame(self, frame_lines: list[str]) -> "Frame":
        """
        Parse frame lines into a Frame object.

        Args:
            frame_lines: list of strings representing the frame data

        Returns:
            Frame object
        """
        pass

    @abstractmethod
    def _parse_trajectory(self, file_index: int):
        """Parse trajectory file at given index, storing frame locations."""
        pass

    def __len__(self) -> int:
        return self._total_frames

    def __iter__(self) -> Iterator["Frame"]:
        """Iterate over all frames lazily."""
        for i in range(self._total_frames):
            yield self.read_frame(i)

    def __getitem__(self, index: int | slice) -> "Frame | list[Frame]":
        """Support indexing and slicing of frames."""
        if isinstance(index, int):
            return self.read_frame(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self._total_frames)
            return self.read_range(start, stop, step)
        else:
            raise TypeError("Index must be int or slice")

    def _open_files(self):
        """Open trajectory files with memory mapping and build global index."""
        self._mms = []

        # Try to load existing indexes first
        if self._load_indexes():
            print("Loaded existing indexes")
            # Still need to open memory-mapped files
            for file_index, fpath in enumerate(self.fpaths):
                fp = open(fpath, "rb")
                fp.seek(0, 2)
                if fp.tell() == 0:
                    raise ValueError(f"File is empty: {fpath}")
                fp.seek(0)
                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                self._mms.append(mm)
            return

        for file_index, fpath in enumerate(self.fpaths):
            # Open file
            fp = open(fpath, "rb")
            try:
                # Check if empty
                fp.seek(0, 2)
                if fp.tell() == 0:
                    fp.close()
                    raise ValueError(f"File is empty: {fpath}")
                fp.seek(0)  # Seek back to beginning

                mm = mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ)
                self._mms.append(mm)

                logger.info(
                    f"Processing file {file_index + 1}/{len(self.fpaths)}: {fpath}"
                )
                self._parse_trajectory(file_index)

                # Save index for this specific file after processing
                self._save_index_for_file(file_index)
                logger.info(f"Index saved for {fpath}")

            finally:
                # Always close the file handle
                fp.close()

    def _get_frame_location(self, index: int) -> FrameLocation:
        """Get location information for a frame."""
        if index >= len(self._frame_locations):
            raise IndexError(f"Frame index {index} out of range")
        return self._frame_locations[index]

    def _get_mmap(self, file_index: int) -> mmap.mmap:
        """Get the memory-mapped file object for a specific file."""
        if file_index >= len(self._mms) or self._mms[file_index] is None:
            raise ValueError(f"File {file_index} is not properly opened")
        return self._mms[file_index]

    @property
    def fpath(self) -> Path:
        """For backward compatibility - returns the first file path."""
        if not self.fpaths:
            raise ValueError("No files available")
        return self.fpaths[0]

    def _get_index_file_paths(self) -> list[Path]:
        """Get the paths for individual index files."""
        index_files = []
        for fpath in self.fpaths:
            index_filename = f"{fpath.stem}_index.json"
            index_files.append(fpath.parent / index_filename)
        return index_files

    def _save_index_for_file(self, file_index: int) -> None:
        """Save frame locations for a specific file to its index file."""
        try:
            # Get frames belonging to this file and create timestep: offset mapping
            frame_offsets = {}
            timestep = 0
            for loc in self._frame_locations:
                if loc.file_index == file_index:
                    frame_offsets[timestep] = loc.byte_offset
                    timestep += 1

            index_data = {
                "trajectory_file": str(self.fpaths[file_index]),
                "total_frames": len(frame_offsets),
                "frame_offsets": frame_offsets,  # timestep: offset dictionary
            }

            index_file = self._index_files[file_index]
            with open(index_file, "w") as f:
                json.dump(index_data, f, indent=2)
        except Exception as e:
            print(
                f"Warning: Failed to save index file for {self.fpaths[file_index]}: {e}"
            )

    def _load_indexes(self) -> bool:
        """
        Load frame locations from individual index files.

        Returns:
            True if all indexes were successfully loaded, False otherwise.
        """
        # Check if all index files exist
        for index_file in self._index_files:
            if not index_file.exists():
                return False

        try:
            self._frame_locations = []
            total_frames = 0

            for file_index, (index_file, fpath) in enumerate(
                zip(self._index_files, self.fpaths)
            ):
                with open(index_file) as f:
                    index_data = json.load(f)

                # Verify that the trajectory file matches
                stored_file = index_data.get("trajectory_file", "")
                current_file = str(fpath)

                if stored_file != current_file:
                    print(
                        f"Warning: Trajectory file path changed for {fpath}, rebuilding indexes"
                    )
                    return False

                # Load frame offsets for this file
                frame_offsets = index_data[
                    "frame_offsets"
                ]  # timestep: offset dictionary

                # Add each frame location to global list (sorted by timestep)
                for timestep in sorted(frame_offsets.keys(), key=int):
                    byte_offset = frame_offsets[timestep]
                    location = FrameLocation(
                        file_index=file_index,
                        byte_offset=int(byte_offset),
                        file_path=fpath,
                    )
                    self._frame_locations.append(location)

                total_frames += index_data["total_frames"]

            self._total_frames = total_frames
            return True

        except Exception as e:
            print(f"Warning: Failed to load index files: {e}, rebuilding indexes")
            return False


class TrajectoryWriter(ABC):
    """Base class for all trajectory file writers."""

    def __init__(self, fpath: str | Path):
        self.fpath = Path(fpath)
        self._fp = open(self.fpath, "w+b")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @abstractmethod
    def write_frame(self, frame: "Frame"):
        """Write a single frame to the file."""
        pass

    def close(self):
        if hasattr(self, "_fp") and self._fp is not None:
            self._fp.close()
            self._fp = None
