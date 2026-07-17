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
│   │   ├── simbox/         # Optional simulation cell
│   │   └── meta/           # Exact-dtype MetaValue entries
│   ├── 1/                  # Frame 1
│   │   ├── blocks/
│   │   ├── simbox/
│   │   └── meta/
│   └── ...
├── trajectory_schema_version  # Attribute: exact schema version (2)
└── n_frames                # Attribute: total number of frames
"""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment, unused-ignore]

from molrs import Frame

from ..data.h5 import frame_to_h5_group, h5_group_to_frame
from .base import BaseTrajectoryReader, PathLike, TrajectoryWriter

TRAJECTORY_SCHEMA_VERSION = 2


def _validate_trajectory_file(h5_file: "h5py.File") -> int:
    if set(h5_file.attrs) != {"trajectory_schema_version", "n_frames"}:
        raise ValueError(
            "HDF5 trajectory requires exactly trajectory_schema_version and "
            "n_frames attributes"
        )
    version = h5_file.attrs["trajectory_schema_version"]
    if not isinstance(version, Integral) or int(version) != TRAJECTORY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported HDF5 trajectory schema {version!r}; "
            f"required schema is {TRAJECTORY_SCHEMA_VERSION}"
        )
    if set(h5_file.keys()) != {"frames"}:
        raise ValueError("HDF5 trajectory must contain exactly the frames group")

    frames_group = h5_file["frames"]
    if not isinstance(frames_group, h5py.Group):
        raise ValueError("HDF5 trajectory frames entry must be a group")
    if frames_group.attrs:
        raise ValueError("HDF5 trajectory frames group has unknown attributes")

    n_frames = h5_file.attrs["n_frames"]
    if not isinstance(n_frames, Integral) or int(n_frames) < 0:
        raise ValueError("HDF5 trajectory n_frames must be a non-negative integer")
    n_frames = int(n_frames)
    expected_names = [str(index) for index in range(n_frames)]
    names = sorted(
        frames_group.keys(), key=lambda name: int(name) if name.isdigit() else -1
    )
    if names != expected_names:
        raise ValueError(
            "HDF5 trajectory frame groups must be contiguous and match n_frames"
        )
    for name in names:
        if not isinstance(frames_group[name], h5py.Group):
            raise ValueError(f"HDF5 trajectory frame {name!r} must be a group")
    return n_frames


def _initialize_trajectory_file(h5_file: "h5py.File") -> None:
    h5_file.attrs["trajectory_schema_version"] = TRAJECTORY_SCHEMA_VERSION
    h5_file.attrs["n_frames"] = 0
    h5_file.create_group("frames")


class HDF5TrajectoryReader(BaseTrajectoryReader):
    """Read Trajectory objects from HDF5 files.

    A binary, random-access :class:`BaseTrajectoryReader`: it implements only
    ``read_frame`` + ``n_frames`` and inherits ``__iter__`` / ``__getitem__`` /
    slicing / ``__len__`` from the pure base (no mmap).

    The HDF5 file structure should follow:
    - /frames/{frame_index}/blocks/ for data blocks
    - /frames/{frame_index}/meta/ for exact-dtype frame metadata
    - /trajectory_schema_version and /n_frames attributes

    Examples:
        >>> reader = HDF5TrajectoryReader("trajectory.h5")
        >>> frame = reader.read_frame(0)
        >>> for frame in reader:
        ...     process(frame)
    """

    def __init__(self, path: PathLike, **open_kwargs):
        """Initialize HDF5 trajectory reader.

        Args:
            path: Path to HDF5 trajectory file.
            **open_kwargs (Any): Additional arguments passed to h5py.File.
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. Install it with: pip install h5py"
            )
        super().__init__(path)  # BaseReader: normalize fpaths + validate existence
        self._path = self.fpath
        self._open_kwargs = open_kwargs
        self._file: h5py.File | None = None
        with h5py.File(self._path, mode="r", **self._open_kwargs) as h5_file:
            self._n_frames = _validate_trajectory_file(h5_file)

    def __enter__(self):
        """Open the HDF5 file and cache the frame count."""
        self._file = h5py.File(self._path, mode="r", **self._open_kwargs)
        try:
            self._n_frames = _validate_trajectory_file(self._file)
        except Exception:
            self._file.close()
            self._file = None
            raise
        return self

    def close(self) -> None:
        """Close the HDF5 file if open."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def n_frames(self) -> int:
        """Number of frames in the trajectory."""
        return self._n_frames

    def _get_n_frames(self) -> int:
        """Get number of frames from open file."""
        if self._file is not None:
            return _validate_trajectory_file(self._file)
        with h5py.File(self._path, "r", **self._open_kwargs) as h5_file:
            return _validate_trajectory_file(h5_file)

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


class HDF5TrajectoryWriter(TrajectoryWriter):
    """Write Trajectory objects to HDF5 files.

    The HDF5 file structure follows:
    - /frames/{frame_index}/blocks/ for data blocks
    - /frames/{frame_index}/meta/ for exact-dtype frame metadata
    - /trajectory_schema_version and /n_frames attributes

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
            **open_kwargs (Any): Additional arguments passed to h5py.File.
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. Install it with: pip install h5py"
            )
        self._path = Path(path)
        self._open_kwargs = open_kwargs
        self.compression = compression
        self.compression_opts = compression_opts
        self._file: h5py.File | None = None
        self._frame_count = 0

        # Existing files must already implement the exact current schema.
        if self._path.exists():
            self._file = h5py.File(self._path, mode="a", **self._open_kwargs)
            try:
                self._frame_count = _validate_trajectory_file(self._file)
            except Exception:
                self._file.close()
                self._file = None
                raise
        else:
            self._file = h5py.File(self._path, mode="w", **self._open_kwargs)
            _initialize_trajectory_file(self._file)

    def __enter__(self):
        """Open HDF5 file."""
        if self._file is None:
            if self._path.exists():
                self._file = h5py.File(self._path, mode="a", **self._open_kwargs)
                try:
                    self._frame_count = _validate_trajectory_file(self._file)
                except Exception:
                    self._file.close()
                    self._file = None
                    raise
            else:
                self._file = h5py.File(self._path, mode="w", **self._open_kwargs)
                _initialize_trajectory_file(self._file)
                self._frame_count = 0
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

        if _validate_trajectory_file(self._file) != self._frame_count:
            raise ValueError("HDF5 trajectory changed while writer was open")
        frames_group = self._file["frames"]

        # Create frame group
        frame_key = str(self._frame_count)
        frame_group = frames_group.create_group(frame_key)
        try:
            frame_to_h5_group(
                frame, frame_group, self.compression, self.compression_opts
            )
        except Exception:
            del frames_group[frame_key]
            raise

        # Update frame count and n_frames attribute
        self._frame_count += 1
        self._file.attrs["n_frames"] = self._frame_count

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None
