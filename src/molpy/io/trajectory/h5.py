"""HDF5 trajectory file format support.

This module provides reading and writing of Trajectory objects to/from HDF5 format
using h5py. The HDF5 format is efficient for storing large trajectory datasets
and supports compression and chunking.

HDF5 Trajectory Structure:
--------------------------
/                           # Root group
├── frames/                 # Group containing all frames
│   ├── 0/                  # Frame 0
│   │   ├── blocks/         # Data blocks (same structure as single Frame)
│   │   └── metadata/       # Frame metadata
│   ├── 1/                  # Frame 1
│   │   ├── blocks/
│   │   └── metadata/
│   └── ...
├── n_frames                # Attribute: total number of frames
└── metadata/               # Optional trajectory-level metadata
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment, unused-ignore]

from molpy.core import Frame

from ..data.h5 import frame_to_h5_group, h5_group_to_frame
from .base import PathLike, TrajectoryWriter


class HDF5TrajectoryReader:
    """Read Trajectory objects from HDF5 files.

    The HDF5 file structure should follow:
    - /frames/{frame_index}/blocks/ for data blocks
    - /frames/{frame_index}/metadata/ for frame metadata
    - /n_frames attribute for total frame count

    Examples:
        >>> reader = HDF5TrajectoryReader("trajectory.h5")
        >>> frame = reader.read_frame(0)
        >>> for frame in reader:
        ...     process(frame)
    """

    def __init__(self, path: PathLike, **open_kwargs):
        """Initialize HDF5 trajectory reader.

        Args:
            path: Path to HDF5 trajectory file
            **open_kwargs: Additional arguments passed to h5py.File
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install it with: pip install h5py"
            )
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"HDF5 trajectory file not found: {self._path}")
        self._open_kwargs = open_kwargs
        self._file: h5py.File | None = None
        self._n_frames: int | None = None

    def __enter__(self):
        """Open HDF5 file."""
        self._file = h5py.File(self._path, mode="r", **self._open_kwargs)
        self._n_frames = self._get_n_frames()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        if self._n_frames is None:
            with h5py.File(self._path, "r") as f:
                self._n_frames = self._get_n_frames_from_file(f)
        return self._n_frames

    def _get_n_frames(self) -> int:
        """Get number of frames from open file."""
        if self._file is None:
            with h5py.File(self._path, "r") as f:
                return self._get_n_frames_from_file(f)
        return self._get_n_frames_from_file(self._file)

    def _get_n_frames_from_file(self, f: h5py.File) -> int:
        """Get number of frames from HDF5 file.

        Args:
            f: Open HDF5 file handle

        Returns:
            Number of frames in the trajectory
        """
        if "frames" not in f:
            return 0

        frames_group = f["frames"]
        # Count frame groups (they should be named "0", "1", "2", ...)
        frame_keys = [k for k in frames_group.keys() if k.isdigit()]
        if frame_keys:
            # Get the maximum frame index and add 1
            max_index = max(int(k) for k in frame_keys)
            # Check if all indices from 0 to max_index exist
            expected_count = max_index + 1
            if all(str(i) in frames_group for i in range(expected_count)):
                return expected_count
            # Otherwise count what we have
            return len(frame_keys)

        # Fallback: check n_frames attribute
        if "n_frames" in f.attrs:
            return int(f.attrs["n_frames"])

        return 0

    def read_frame(self, index: int) -> Frame:
        """Read a specific frame from the trajectory.

        Args:
            index: Frame index (0-based)

        Returns:
            Frame object

        Raises:
            IndexError: If index is out of range
        """
        if index < 0:
            index = self.n_frames + index

        if index < 0 or index >= self.n_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self.n_frames})")

        # Use context manager if file is not open
        if self._file is None:
            with h5py.File(self._path, "r") as f:
                return self._read_frame_from_file(f, index)
        else:
            return self._read_frame_from_file(self._file, index)

    def _read_frame_from_file(self, f: h5py.File, index: int) -> Frame:
        """Read frame from open HDF5 file.

        Args:
            f: Open HDF5 file handle
            index: Frame index

        Returns:
            Frame object
        """
        frames_group = f["frames"]
        frame_key = str(index)

        if frame_key not in frames_group:
            raise IndexError(f"Frame {index} not found in HDF5 file")

        frame_group = frames_group[frame_key]
        return h5_group_to_frame(frame_group)

    def __iter__(self) -> Iterator[Frame]:
        """Iterate over all frames lazily."""
        if self.n_frames == 0:
            return
        with h5py.File(self._path, "r") as f:
            if "frames" not in f:
                return
            frames_group = f["frames"]
            # Get sorted frame indices
            frame_indices = sorted(
                (int(k) for k in frames_group.keys() if k.isdigit()),
                key=int,
            )
            for index in frame_indices:
                frame_group = frames_group[str(index)]
                yield h5_group_to_frame(frame_group)

    def __len__(self) -> int:
        """Return number of frames."""
        return self.n_frames

    def __getitem__(self, index: int | slice) -> Frame | list[Frame]:
        """Support indexing and slicing of frames.

        Args:
            index: Frame index or slice

        Returns:
            Frame or list of Frames
        """
        if isinstance(index, int):
            return self.read_frame(index)
        elif isinstance(index, slice):
            start, stop, step = index.indices(self.n_frames)
            return [self.read_frame(i) for i in range(start, stop, step)]
        else:
            raise TypeError("Index must be int or slice")


class HDF5TrajectoryWriter(TrajectoryWriter):
    """Write Trajectory objects to HDF5 files.

    The HDF5 file structure follows:
    - /frames/{frame_index}/blocks/ for data blocks
    - /frames/{frame_index}/metadata/ for frame metadata
    - /n_frames attribute for total frame count

    Examples:
        >>> writer = HDF5TrajectoryWriter("trajectory.h5")
        >>> writer.write_frame(frame0)
        >>> writer.write_frame(frame1)
        >>> writer.close()
    """

    def __init__(
        self,
        path: PathLike,
        compression: str | None = "gzip",
        compression_opts: int = 4,
        **open_kwargs,
    ):
        """Initialize HDF5 trajectory writer.

        Args:
            path: Path to output HDF5 file
            compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
                Defaults to 'gzip'.
            compression_opts: Compression level (for gzip: 0-9). Defaults to 4.
            **open_kwargs: Additional arguments passed to h5py.File
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install it with: pip install h5py"
            )
        self._path = Path(path)
        self._open_kwargs = open_kwargs
        self.compression = compression
        self.compression_opts = compression_opts
        self._file: h5py.File | None = None
        self._frame_count = 0

        # Open file in append mode if it exists, otherwise create new
        if self._path.exists():
            self._file = h5py.File(self._path, mode="a", **self._open_kwargs)
            # Get current frame count
            if "frames" in self._file:
                frames_group = self._file["frames"]
                frame_keys = [k for k in frames_group.keys() if k.isdigit()]
                if frame_keys:
                    self._frame_count = max(int(k) for k in frame_keys) + 1
        else:
            self._file = h5py.File(self._path, mode="w", **self._open_kwargs)

    def __enter__(self):
        """Open HDF5 file."""
        if self._file is None:
            if self._path.exists():
                self._file = h5py.File(self._path, mode="a", **self._open_kwargs)
            else:
                self._file = h5py.File(self._path, mode="w", **self._open_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        self.close()

    def write_frame(self, frame: Frame) -> None:
        """Write a single frame to the trajectory file.

        Args:
            frame: Frame object to write
        """
        if self._file is None:
            raise ValueError("File not open. Use 'with' statement or call __enter__")

        # Create or get frames group
        if "frames" not in self._file:
            frames_group = self._file.create_group("frames")
        else:
            frames_group = self._file["frames"]

        # Create frame group
        frame_key = str(self._frame_count)
        frame_group = frames_group.create_group(frame_key)

        # Write frame using conversion function
        frame_to_h5_group(frame, frame_group, self.compression, self.compression_opts)

        # Update frame count and n_frames attribute
        self._frame_count += 1
        self._file.attrs["n_frames"] = self._frame_count

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
