"""Zarr backend for SimStore.

Targets zarr-python >= 3.0. Minimizes small files by:
- Using single-chunk storage for small arrays (frame, forcefield)
- Using consolidate_metadata() to merge .zarray/.zgroup files
- Storing string data as attributes to avoid codec issues
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import zarr
except ImportError:
    zarr = None  # type: ignore[assignment]


class ZarrBackend:
    """Thin wrapper around zarr.Group providing a normalized API."""

    def __init__(self, path: str, mode: str = "r"):
        if zarr is None:
            raise ImportError(
                "zarr >= 3.0 is required for Zarr store support. "
                "Install with: pip install 'zarr>=3.0'"
            )
        self._path = Path(path)
        self._mode = mode
        self._root = zarr.open_group(str(path), mode=mode)
        self._group = self._root

    @classmethod
    def _wrap(cls, group) -> "ZarrBackend":
        obj = object.__new__(cls)
        obj._path = None
        obj._mode = None
        obj._root = None
        obj._group = group
        return obj

    # ── group operations ──────────────────────────────────────────────

    def create_group(self, name: str) -> "ZarrBackend":
        if name in self._group:
            g = self._group[name]
        else:
            g = self._group.create_group(name)
        return self._wrap(g)

    def has(self, name: str) -> bool:
        return name in self._group

    def __contains__(self, name: str) -> bool:
        return name in self._group

    def __getitem__(self, name: str) -> "ZarrBackend":
        item = self._group[name]
        if isinstance(item, zarr.Group):
            return self._wrap(item)
        raise KeyError(f"{name} is an array, not a group")

    def list_members(self) -> list[str]:
        return list(self._group.keys())

    def list_arrays(self) -> list[str]:
        """List only array members, not sub-groups."""
        result = []
        for k in self._group.keys():
            item = self._group[k]
            if isinstance(item, zarr.Array):
                result.append(k)
        # Also include string data stored as _str_* attributes
        for k in dict(self._group.attrs).keys():
            if k.startswith("_str_"):
                result.append(k[5:])  # strip _str_ prefix
        return result

    def list_groups(self) -> list[str]:
        return [k for k in self._group.keys() if isinstance(self._group[k], zarr.Group)]

    # ── write operations ──────────────────────────────────────────────

    def write_array(
        self,
        name: str,
        data: np.ndarray,
        chunks: tuple | None = None,
        **kwargs,
    ) -> None:
        data = np.asarray(data)

        if data.dtype.kind == "U":
            # Store string data as a JSON-encoded attribute to avoid codec issues.
            # Prefix with _str_ so read_array can find it.
            if data.ndim == 0:
                self._group.attrs[f"_str_{name}"] = str(data)
            else:
                self._group.attrs[f"_str_{name}"] = json.dumps(
                    [str(s) for s in data.flat]
                )
            # Also store shape for reconstruction
            self._group.attrs[f"_str_{name}_shape"] = list(data.shape)
            return

        if chunks is None:
            # Single chunk → one file per array → minimal file count
            chunks = data.shape if data.size > 0 else None

        if data.size == 0:
            self._group.create_array(
                name, shape=data.shape, dtype=data.dtype, overwrite=True
            )
        else:
            self._group.create_array(name, data=data, chunks=chunks, overwrite=True)

    def write_attr(self, name: str, value) -> None:
        self._group.attrs[name] = value

    # ── read operations ───────────────────────────────────────────────

    def read_array(self, name: str) -> np.ndarray:
        # Check if it's a string stored as attribute
        str_key = f"_str_{name}"
        attrs = dict(self._group.attrs)
        if str_key in attrs:
            val = attrs[str_key]
            shape_key = f"_str_{name}_shape"
            if isinstance(val, str):
                # Could be a JSON list or a scalar string
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        shape = tuple(attrs.get(shape_key, [len(parsed)]))
                        return np.array(parsed).reshape(shape)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Scalar string
                return np.array(val)
            return np.asarray(val)

        arr = self._group[name]
        return np.asarray(arr)

    def read_attr(self, name: str):
        return self._group.attrs[name]

    def has_attr(self, name: str) -> bool:
        return name in self._group.attrs

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        if self._root is not None and self._mode in ("w", "a"):
            try:
                zarr.consolidate_metadata(self._root.store)
            except Exception:
                pass

    def flush(self) -> None:
        pass
