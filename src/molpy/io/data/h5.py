"""HDF5 file format support for Frame objects.

This module provides reading and writing of Frame objects to/from HDF5 format
using h5py. The HDF5 format is efficient for storing large molecular datasets
and supports compression and chunking.

HDF5 Structure:
---------------
/                           # Root group
├── blocks/                  # Group containing all data blocks
│   ├── atoms/              # Block group (e.g., "atoms")
│   │   ├── x               # Dataset (variable)
│   │   ├── y               # Dataset
│   │   └── z               # Dataset
│   └── bonds/              # Another block group
│       ├── i               # Dataset
│       └── j               # Dataset
└── metadata/               # Group containing metadata
    ├── timestep            # Attribute or dataset
    └── ...                 # Other metadata
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment, unused-ignore]

from molpy.core import Block, Frame

try:
    from molpy.core import Box
except ImportError:
    Box = None  # type: ignore[assignment, unused-ignore]

from .base import PathLike


# =============================================================================
# Frame <-> HDF5 Conversion Functions (reusable for trajectory)
# =============================================================================


def frame_to_h5_group(
    frame: Frame,
    h5_group: "h5py.Group",
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """Write a Frame to an HDF5 group.

    This function can be used to write a Frame to any HDF5 group, making it
    reusable for both single Frame files and trajectory files.

    Args:
        frame: Frame object to write
        h5_group: HDF5 group to write to
        compression: Compression algorithm (None, 'gzip', 'lzf', 'szip').
            Defaults to 'gzip'.
        compression_opts: Compression level (for gzip: 0-9). Defaults to 4.

    Raises:
        ValueError: If frame is empty (no blocks).
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 support. " "Install it with: pip install h5py"
        )

    if not frame._blocks:
        raise ValueError("Cannot write empty frame (no blocks)")

    # Write blocks
    blocks_group = h5_group.create_group("blocks")
    for block_name, block in frame._blocks.items():
        block_group = blocks_group.create_group(block_name)

        # Write each variable in the block
        for var_name, var_data in block._vars.items():
            # Ensure data is a numpy array
            data = np.asarray(var_data)

            # Handle string types - h5py doesn't support Unicode arrays directly
            if data.dtype.kind == "U":  # Unicode string
                # Convert to variable-length UTF-8 strings
                # h5py supports variable-length strings via object dtype
                data_as_objects = data.astype(object)
                # lzf doesn't support compression_opts
                create_kwargs = {
                    "compression": compression,
                    "shuffle": True if compression else False,
                }
                if compression == "gzip" and compression_opts is not None:
                    create_kwargs["compression_opts"] = compression_opts
                elif compression == "lzf":
                    # lzf doesn't use compression_opts
                    pass

                dataset = block_group.create_dataset(
                    var_name,
                    data=data_as_objects,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    **create_kwargs,
                )
            else:
                # Create dataset with compression for numeric types
                # lzf doesn't support compression_opts
                create_kwargs = {
                    "compression": compression,
                    "shuffle": True if compression else False,
                }
                if compression == "gzip" and compression_opts is not None:
                    create_kwargs["compression_opts"] = compression_opts
                elif compression == "lzf":
                    # lzf doesn't use compression_opts
                    pass

                dataset = block_group.create_dataset(
                    var_name,
                    data=data,
                    **create_kwargs,
                )

            # Store dtype information as attribute for better reconstruction
            dataset.attrs["dtype"] = str(data.dtype)

    # Write metadata
    if frame.metadata:
        metadata_group = h5_group.create_group("metadata")
        _write_metadata_to_group(
            metadata_group, frame.metadata, compression, compression_opts
        )


def h5_group_to_frame(h5_group: "h5py.Group", frame: Frame | None = None) -> Frame:
    """Read a Frame from an HDF5 group.

    This function can be used to read a Frame from any HDF5 group, making it
    reusable for both single Frame files and trajectory files.

    Args:
        h5_group: HDF5 group to read from
        frame: Optional existing Frame to populate. If None, creates a new one.

    Returns:
        Frame: Populated Frame object with blocks and metadata from HDF5 group.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 support. " "Install it with: pip install h5py"
        )

    frame = frame or Frame()

    # Read blocks
    if "blocks" in h5_group:
        blocks_group = h5_group["blocks"]
        for block_name in blocks_group.keys():
            block_group = blocks_group[block_name]
            block = Block()

            # Read all variables in this block
            for var_name in block_group.keys():
                dataset = block_group[var_name]

                # Restore original dtype if it was a Unicode string
                if "dtype" in dataset.attrs:
                    original_dtype_str = dataset.attrs["dtype"]
                    if original_dtype_str.startswith(
                        "<U"
                    ) or original_dtype_str.startswith(">U"):
                        # Use asstr() method for variable-length strings
                        try:
                            data = dataset.asstr()[:]
                            # Convert to numpy array with original dtype
                            data = np.array(data, dtype=original_dtype_str)
                        except (AttributeError, TypeError):
                            # Fallback: read as array and decode
                            raw_data = np.array(dataset)
                            if raw_data.dtype.kind == "O":  # Object dtype
                                data = np.array(
                                    [
                                        (
                                            s.decode("utf-8")
                                            if isinstance(s, bytes)
                                            else str(s)
                                        )
                                        for s in raw_data
                                    ],
                                    dtype=original_dtype_str,
                                )
                            else:
                                data = raw_data.astype(original_dtype_str)
                    else:
                        # Read dataset as numpy array for non-string types
                        data = np.array(dataset)
                else:
                    # Read dataset as numpy array
                    data = np.array(dataset)

                block[var_name] = data

            frame[block_name] = block

    # Read metadata
    if "metadata" in h5_group:
        metadata_group = h5_group["metadata"]
        _read_metadata_from_group(metadata_group, frame.metadata)

    return frame


def _write_metadata_to_group(
    metadata_group: "h5py.Group",
    metadata_dict: dict,
    compression: str | None = "gzip",
    compression_opts: int = 4,
) -> None:
    """Recursively write metadata to HDF5 group.

    Args:
        metadata_group: HDF5 group to write metadata to
        metadata_dict: Dictionary containing metadata
        compression: Compression algorithm
        compression_opts: Compression level
    """
    for key, value in metadata_dict.items():
        # Skip None values
        if value is None:
            continue

        # Handle Box objects specially
        if Box is not None and isinstance(value, Box):
            # Store Box as a group with matrix, pbc, and origin
            box_group = metadata_group.create_group(key)
            # Use np.array(box) which will call __array__ method
            box_matrix = np.array(value)
            # lzf doesn't support compression_opts
            create_kwargs = {
                "compression": compression,
            }
            if compression == "gzip" and compression_opts is not None:
                create_kwargs["compression_opts"] = compression_opts
            elif compression == "lzf":
                # lzf doesn't use compression_opts
                pass

            box_group.create_dataset("matrix", data=box_matrix, **create_kwargs)
            box_group.create_dataset("pbc", data=value.pbc, **create_kwargs)
            box_group.create_dataset("origin", data=value.origin, **create_kwargs)
            box_group.attrs["_type"] = "Box"
            continue

        # Handle different types
        if isinstance(value, (str, int, float, bool)):
            # Store simple types as attributes
            metadata_group.attrs[key] = value
        elif isinstance(value, np.ndarray):
            # Store numpy arrays as datasets
            # lzf doesn't support compression_opts
            create_kwargs = {
                "compression": compression,
            }
            if compression == "gzip" and compression_opts is not None:
                create_kwargs["compression_opts"] = compression_opts
            elif compression == "lzf":
                # lzf doesn't use compression_opts
                pass

            metadata_group.create_dataset(key, data=value, **create_kwargs)
        elif isinstance(value, (list, tuple)):
            # Convert lists/tuples to numpy arrays
            try:
                arr = np.asarray(value)
                # lzf doesn't support compression_opts
                create_kwargs = {
                    "compression": compression,
                }
                if compression == "gzip" and compression_opts is not None:
                    create_kwargs["compression_opts"] = compression_opts
                elif compression == "lzf":
                    # lzf doesn't use compression_opts
                    pass

                metadata_group.create_dataset(key, data=arr, **create_kwargs)
            except (ValueError, TypeError):
                # If conversion fails, store as JSON string
                metadata_group.attrs[key] = json.dumps(value)
        elif isinstance(value, dict):
            # Recursively write nested dictionaries as groups
            nested_group = metadata_group.create_group(key)
            _write_metadata_to_group(nested_group, value, compression, compression_opts)
        else:
            # For other types, try to serialize as JSON
            try:
                json_str = json.dumps(value)
                metadata_group.attrs[key] = json_str
            except (TypeError, ValueError):
                # If JSON serialization fails, convert to string
                metadata_group.attrs[key] = str(value)


def _read_metadata_from_group(
    metadata_group: "h5py.Group", metadata_dict: dict
) -> None:
    """Recursively read metadata from HDF5 group.

    Args:
        metadata_group: HDF5 group containing metadata
        metadata_dict: Dictionary to populate with metadata
    """
    for key in metadata_group.keys():
        item = metadata_group[key]

        if isinstance(item, h5py.Dataset):
            # Read dataset
            data = np.array(item)
            # Convert scalar arrays to Python types
            if data.shape == ():
                metadata_dict[key] = data.item()
            elif data.dtype.kind == "U":  # Unicode string
                if data.shape == ():
                    metadata_dict[key] = str(data.item())
                else:
                    metadata_dict[key] = [str(s) for s in data]
            else:
                metadata_dict[key] = data.tolist() if data.size < 1000 else data
        elif isinstance(item, h5py.Group):
            # Check if this is a Box object
            if (
                "_type" in item.attrs
                and item.attrs["_type"] == "Box"
                and Box is not None
            ):
                # Reconstruct Box object
                matrix = np.array(item["matrix"])
                pbc = np.array(item["pbc"])
                origin = np.array(item["origin"])
                metadata_dict[key] = Box(matrix=matrix, pbc=pbc, origin=origin)
            else:
                # Recursively read nested groups
                nested_dict = {}
                _read_metadata_from_group(item, nested_dict)
                metadata_dict[key] = nested_dict

    # Also read attributes
    for attr_name in metadata_group.attrs.keys():
        attr_value = metadata_group.attrs[attr_name]
        # Skip internal type markers
        if attr_name == "_type":
            continue

        # Convert numpy types to Python types
        if isinstance(attr_value, np.ndarray):
            if attr_value.shape == ():
                metadata_dict[attr_name] = attr_value.item()
            elif attr_value.dtype.kind == "U":  # Unicode string
                if attr_value.shape == ():
                    metadata_dict[attr_name] = str(attr_value.item())
                else:
                    metadata_dict[attr_name] = [str(s) for s in attr_value]
            else:
                metadata_dict[attr_name] = (
                    attr_value.tolist() if attr_value.size < 1000 else attr_value
                )
        elif isinstance(attr_value, (bytes, str)):
            # Try to decode JSON strings
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode("utf-8")
            try:
                metadata_dict[attr_name] = json.loads(attr_value)
            except (json.JSONDecodeError, TypeError):
                metadata_dict[attr_name] = attr_value
        else:
            metadata_dict[attr_name] = attr_value


class HDF5Reader:
    """Read Frame objects from HDF5 files.

    The HDF5 file structure should follow the format:
    - /blocks/{block_name}/{variable_name} for data arrays
    - /metadata/ for frame metadata

    Examples:
        >>> reader = HDF5Reader("frame.h5")
        >>> frame = reader.read()
        >>> frame["atoms"]["x"]
        array([0., 1., 2.])
    """

    def __init__(self, path: PathLike, **open_kwargs):
        """Initialize HDF5 reader.

        Args:
            path: Path to HDF5 file
            **open_kwargs: Additional arguments passed to h5py.File
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. "
                "Install it with: pip install h5py"
            )
        self._path = Path(path)
        self._open_kwargs = open_kwargs
        self._file: h5py.File | None = None

    def __enter__(self):
        """Open HDF5 file."""
        self._file = h5py.File(self._path, mode="r", **self._open_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None

    def read(self, frame: Frame | None = None) -> Frame:
        """Read Frame from HDF5 file.

        Args:
            frame: Optional existing Frame to populate. If None, creates a new one.

        Returns:
            Frame: Populated Frame object with blocks and metadata from HDF5 file.
        """
        with h5py.File(self._path, "r") as f:
            return h5_group_to_frame(f, frame)


class HDF5Writer:
    """Write Frame objects to HDF5 files.

    The HDF5 file structure follows:
    - /blocks/{block_name}/{variable_name} for data arrays
    - /metadata/ for frame metadata

    Examples:
        >>> frame = Frame(blocks={"atoms": {"x": [0, 1, 2], "y": [0, 0, 0]}})
        >>> writer = HDF5Writer("frame.h5")
        >>> writer.write(frame)
    """

    def __init__(
        self,
        path: PathLike,
        compression: str | None = "gzip",
        compression_opts: int = 4,
        **open_kwargs,
    ):
        """Initialize HDF5 writer.

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
        self._file: h5py.File | None = None
        self.compression = compression
        self.compression_opts = compression_opts

    def __enter__(self):
        """Open HDF5 file."""
        self._file = h5py.File(self._path, mode="w", **self._open_kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None

    def write(self, frame: Frame) -> None:
        """Write Frame to HDF5 file.

        Args:
            frame: Frame object to write.

        Raises:
            ValueError: If frame is empty (no blocks).
        """
        # If file is already open (context manager), use it
        if self._file is not None:
            frame_to_h5_group(
                frame, self._file, self.compression, self.compression_opts
            )
        else:
            # Otherwise, open and close file
            with h5py.File(self._path, "w") as f:
                frame_to_h5_group(frame, f, self.compression, self.compression_opts)

