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
├── simbox/                 # Optional simulation cell
└── meta/                   # Exact-dtype MetaValue entries (schema v2)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment, unused-ignore]

from molrs import Block, Frame, MetaValue

from .base import PathLike

FRAME_SCHEMA_VERSION = 2

_META_DTYPES: dict[str, tuple[np.dtype | None, tuple[int, ...]]] = {
    "bool": (np.dtype(np.bool_), ()),
    "i32": (np.dtype(np.int32), ()),
    "i64": (np.dtype(np.int64), ()),
    "u32": (np.dtype(np.uint32), ()),
    "u64": (np.dtype(np.uint64), ()),
    "f32": (np.dtype(np.float32), ()),
    "f64": (np.dtype(np.float64), ()),
    "string": (None, ()),
    "bool3": (np.dtype(np.bool_), (3,)),
    "i32x3": (np.dtype(np.int32), (3,)),
    "i64x3": (np.dtype(np.int64), (3,)),
    "u32x3": (np.dtype(np.uint32), (3,)),
    "u64x3": (np.dtype(np.uint64), (3,)),
    "f32x3": (np.dtype(np.float32), (3,)),
    "f64x3": (np.dtype(np.float64), (3,)),
    "f32x6": (np.dtype(np.float32), (6,)),
    "f64x6": (np.dtype(np.float64), (6,)),
    "f32x9": (np.dtype(np.float32), (9,)),
    "f64x9": (np.dtype(np.float64), (9,)),
}

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
            "h5py is required for HDF5 support. Install it with: pip install h5py"
        )

    if len(frame) == 0:
        raise ValueError("Cannot write empty frame (no blocks)")

    h5_group.attrs["frame_schema_version"] = FRAME_SCHEMA_VERSION

    # Write blocks
    blocks_group = h5_group.create_group("blocks")
    for block_name, block in frame._blocks.items():
        block_group = blocks_group.create_group(block_name)

        # Write each variable in the block
        for var_name, var_data in block._as_dict().items():
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

    # Write the simulation cell as a dedicated group.
    if frame.simbox is not None:
        box_group = h5_group.create_group("simbox")
        box_group.create_dataset(
            "matrix", data=np.asarray(frame.simbox.matrix, dtype=np.float64)
        )
        box_group.create_dataset(
            "origin", data=np.asarray(frame.simbox.origin, dtype=np.float64)
        )
        box_group.create_dataset("pbc", data=np.asarray(frame.simbox.pbc, dtype=bool))

    meta_group = h5_group.create_group("meta")
    _write_typed_meta(meta_group, frame.meta)


def h5_group_to_frame(h5_group: "h5py.Group", frame: Frame | None = None) -> Frame:
    """Read a Frame from an HDF5 group.

    This function can be used to read a Frame from any HDF5 group, making it
    reusable for both single Frame files and trajectory files.

    Args:
        h5_group: HDF5 group to read from
        frame: Optional existing Frame to populate. If None, creates a new one.

    Returns:
        Frame: Populated Frame object with blocks and typed metadata.
    """
    if h5py is None:
        raise ImportError(
            "h5py is required for HDF5 support. Install it with: pip install h5py"
        )

    _validate_frame_group(h5_group)
    frame = frame if frame is not None else Frame()

    # Read blocks
    blocks_group = h5_group["blocks"]
    for block_name in blocks_group.keys():
        block_group = blocks_group[block_name]
        if not isinstance(block_group, h5py.Group):
            raise ValueError(f"HDF5 block {block_name!r} must be a group")
        if block_group.attrs:
            raise ValueError(f"HDF5 block {block_name!r} has unknown attributes")

        block = Block()
        for var_name in block_group.keys():
            dataset = block_group[var_name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(
                    f"HDF5 block variable {block_name!r}/{var_name!r} must be a dataset"
                )
            if set(dataset.attrs) != {"dtype"}:
                raise ValueError(
                    f"HDF5 block variable {block_name!r}/{var_name!r} "
                    "must have exactly one 'dtype' attribute"
                )

            original_dtype_str = _read_string_attr(dataset, "dtype")
            try:
                original_dtype = np.dtype(original_dtype_str)
            except TypeError as exc:
                raise ValueError(
                    f"Invalid NumPy dtype {original_dtype_str!r} for "
                    f"{block_name!r}/{var_name!r}"
                ) from exc

            if original_dtype.kind == "U":
                string_info = h5py.check_string_dtype(dataset.dtype)
                if string_info is None or string_info.encoding != "utf-8":
                    raise ValueError(
                        f"HDF5 block variable {block_name!r}/{var_name!r} "
                        "must use UTF-8 storage"
                    )
                data = np.asarray(dataset.asstr()[...], dtype=original_dtype)
            else:
                if dataset.dtype != original_dtype:
                    raise ValueError(
                        f"HDF5 block variable {block_name!r}/{var_name!r} "
                        f"declares {original_dtype} but stores {dataset.dtype}"
                    )
                data = np.asarray(dataset[...], dtype=original_dtype)

            block[var_name] = data

        frame[block_name] = block

    if "simbox" in h5_group:
        from molpy.core.box import Box

        box_grp = h5_group["simbox"]
        _validate_simbox_group(box_grp)
        frame.simbox = Box(
            matrix=np.array(box_grp["matrix"]),
            origin=np.array(box_grp["origin"]),
            pbc=np.array(box_grp["pbc"]),
        )

    frame.meta = _read_typed_meta(h5_group["meta"])

    return frame


def _validate_frame_group(h5_group: "h5py.Group") -> None:
    if set(h5_group.attrs) != {"frame_schema_version"}:
        raise ValueError(
            "HDF5 Frame requires exactly the 'frame_schema_version' attribute"
        )
    version = h5_group.attrs["frame_schema_version"]
    if (
        not isinstance(version, (int, np.integer))
        or int(version) != FRAME_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported HDF5 Frame schema {version!r}; "
            f"required schema is {FRAME_SCHEMA_VERSION}"
        )

    groups = set(h5_group.keys())
    required = {"blocks", "meta"}
    allowed = required | {"simbox"}
    if missing := required - groups:
        raise ValueError(f"HDF5 Frame is missing groups: {sorted(missing)}")
    if unknown := groups - allowed:
        raise ValueError(f"HDF5 Frame has unknown groups: {sorted(unknown)}")
    for name in groups:
        if not isinstance(h5_group[name], h5py.Group):
            raise ValueError(f"HDF5 Frame entry {name!r} must be a group")


def _validate_simbox_group(box_group: "h5py.Group") -> None:
    if box_group.attrs:
        raise ValueError("HDF5 simbox group has unknown attributes")
    if set(box_group.keys()) != {"matrix", "origin", "pbc"}:
        raise ValueError("HDF5 simbox must contain exactly matrix, origin, and pbc")

    expected = {
        "matrix": (np.dtype(np.float64), (3, 3)),
        "origin": (np.dtype(np.float64), (3,)),
        "pbc": (np.dtype(np.bool_), (3,)),
    }
    for name, (dtype, shape) in expected.items():
        dataset = box_group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"HDF5 simbox/{name} must be a dataset")
        if dataset.attrs:
            raise ValueError(f"HDF5 simbox/{name} has unknown attributes")
        if dataset.dtype != dtype or dataset.shape != shape:
            raise ValueError(
                f"HDF5 simbox/{name} must have dtype {dtype} and shape {shape}; "
                f"got dtype {dataset.dtype} and shape {dataset.shape}"
            )


def _write_typed_meta(meta_group: "h5py.Group", meta: dict[str, MetaValue]) -> None:
    meta_group.attrs["schema_version"] = FRAME_SCHEMA_VERSION
    for index, (key, entry) in enumerate(sorted(meta.items())):
        if not isinstance(key, str):
            raise TypeError("Frame meta keys must be strings")
        if not isinstance(entry, MetaValue):
            raise TypeError(f"Frame meta {key!r} must be a molrs.MetaValue")

        dtype_name = entry.dtype
        if dtype_name not in _META_DTYPES:
            raise ValueError(f"Frame meta {key!r} has unknown dtype {dtype_name!r}")
        expected_dtype, expected_shape = _META_DTYPES[dtype_name]

        dataset_name = f"{index:08d}"
        if dtype_name == "string":
            if not isinstance(entry.value, str):
                raise TypeError(f"Frame meta {key!r} string value is not str")
            dataset = meta_group.create_dataset(
                dataset_name,
                data=entry.value,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
        else:
            data = np.asarray(entry.value, dtype=expected_dtype)
            if data.shape != expected_shape:
                raise ValueError(
                    f"Frame meta {key!r} dtype {dtype_name!r} requires shape "
                    f"{expected_shape}, got {data.shape}"
                )
            dataset = meta_group.create_dataset(dataset_name, data=data)

        dataset.attrs["key"] = key
        dataset.attrs["dtype"] = dtype_name


def _read_typed_meta(meta_group: "h5py.Group") -> dict[str, MetaValue]:
    if set(meta_group.attrs) != {"schema_version"}:
        raise ValueError(
            "HDF5 Frame meta requires exactly the 'schema_version' attribute"
        )
    version = meta_group.attrs["schema_version"]
    if (
        not isinstance(version, (int, np.integer))
        or int(version) != FRAME_SCHEMA_VERSION
    ):
        raise ValueError(
            f"Unsupported HDF5 Frame meta schema {version!r}; "
            f"required schema is {FRAME_SCHEMA_VERSION}"
        )

    names = sorted(meta_group.keys())
    expected_names = [f"{index:08d}" for index in range(len(names))]
    if names != expected_names:
        raise ValueError("HDF5 Frame meta entries must use contiguous numeric names")

    meta: dict[str, MetaValue] = {}
    for name in names:
        dataset = meta_group[name]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"HDF5 Frame meta entry {name!r} must be a dataset")
        if set(dataset.attrs) != {"key", "dtype"}:
            raise ValueError(
                f"HDF5 Frame meta entry {name!r} must have exactly key and dtype attrs"
            )

        key = _read_string_attr(dataset, "key")
        if key in meta:
            raise ValueError(f"Duplicate HDF5 Frame meta key {key!r}")
        dtype_name = _read_string_attr(dataset, "dtype")
        if dtype_name not in _META_DTYPES:
            raise ValueError(
                f"HDF5 Frame meta {key!r} has unknown dtype {dtype_name!r}"
            )

        expected_dtype, expected_shape = _META_DTYPES[dtype_name]
        if dataset.shape != expected_shape:
            raise ValueError(
                f"HDF5 Frame meta {key!r} dtype {dtype_name!r} requires shape "
                f"{expected_shape}, got {dataset.shape}"
            )

        if dtype_name == "string":
            string_info = h5py.check_string_dtype(dataset.dtype)
            if string_info is None or string_info.encoding != "utf-8":
                raise ValueError(
                    f"HDF5 Frame meta {key!r} must use UTF-8 string storage"
                )
            value = dataset.asstr()[()]
            if not isinstance(value, str):
                raise ValueError(f"HDF5 Frame meta {key!r} is not a scalar string")
        else:
            if dataset.dtype != expected_dtype:
                raise ValueError(
                    f"HDF5 Frame meta {key!r} dtype {dtype_name!r} must store "
                    f"{expected_dtype}, got {dataset.dtype}"
                )
            data = np.asarray(dataset[...], dtype=expected_dtype)
            value = data.item() if expected_shape == () else data.tolist()

        meta[key] = MetaValue(dtype_name, value)
    return meta


def _read_string_attr(dataset: "h5py.Dataset", name: str) -> str:
    value = dataset.attrs[name]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="strict")
    if not isinstance(value, str):
        raise ValueError(f"HDF5 attribute {name!r} must be a UTF-8 string")
    return value


class HDF5Reader:
    """Read Frame objects from HDF5 files.

    The HDF5 file structure should follow the format:
    - /blocks/{block_name}/{variable_name} for data arrays
    - /simbox/ for the optional simulation cell
    - /meta/ for exact-dtype frame metadata

    Examples:
        >>> reader = HDF5Reader("frame.h5")
        >>> frame = reader.read()
        >>> frame["atoms"]["x"]
        array([0., 1., 2.])
    """

    def __init__(self, path: PathLike, **open_kwargs: Any):
        """Initialize HDF5 reader.

        Args:
            path: Path to HDF5 file
            **open_kwargs: Additional arguments passed to h5py.File
        """
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 support. Install it with: pip install h5py"
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
            Frame: Populated Frame object with blocks and typed metadata.
        """
        with h5py.File(self._path, "r") as f:
            return h5_group_to_frame(f, frame)


class HDF5Writer:
    """Write Frame objects to HDF5 files.

    The HDF5 file structure follows:
    - /blocks/{block_name}/{variable_name} for data arrays
    - /simbox/ for the optional simulation cell
    - /meta/ for exact-dtype frame metadata

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
        **open_kwargs: Any,
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
                "h5py is required for HDF5 support. Install it with: pip install h5py"
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
