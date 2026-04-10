"""h5py backend for SimStore."""

from __future__ import annotations

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None  # type: ignore[assignment]


class H5Backend:
    """Thin wrapper around h5py.Group providing a normalized API."""

    def __init__(self, path: str, mode: str = "r"):
        if h5py is None:
            raise ImportError(
                "h5py is required for HDF5 store support. "
                "Install with: pip install h5py"
            )
        self._file = h5py.File(path, mode)
        self._group: h5py.Group = self._file

    @classmethod
    def _wrap(cls, group: "h5py.Group", file_ref: "h5py.File") -> "H5Backend":
        obj = object.__new__(cls)
        obj._file = file_ref
        obj._group = group
        return obj

    # ── group operations ──────────────────────────────────────────────

    def create_group(self, name: str) -> "H5Backend":
        g = self._group.require_group(name)
        return self._wrap(g, self._file)

    def has(self, name: str) -> bool:
        return name in self._group

    def __contains__(self, name: str) -> bool:
        return name in self._group

    def __getitem__(self, name: str) -> "H5Backend":
        item = self._group[name]
        if isinstance(item, h5py.Group):
            return self._wrap(item, self._file)
        raise KeyError(f"{name} is a dataset, not a group")

    def list_members(self) -> list[str]:
        return list(self._group.keys())

    def list_arrays(self) -> list[str]:
        """List only dataset (array) members, not sub-groups."""
        return [
            k for k in self._group.keys() if isinstance(self._group[k], h5py.Dataset)
        ]

    def list_groups(self) -> list[str]:
        return [k for k in self._group.keys() if isinstance(self._group[k], h5py.Group)]

    # ── write operations ──────────────────────────────────────────────

    def write_array(
        self,
        name: str,
        data: np.ndarray,
        chunks: tuple | None = None,
        compression: str | None = "gzip",
        compression_opts: int = 4,
    ) -> None:
        data = np.asarray(data)

        if data.dtype.kind == "U":
            self._group.create_dataset(
                name,
                data=data.astype(object),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            return

        kw: dict = {}
        if compression and data.size > 0:
            kw["compression"] = compression
            if compression == "gzip" and compression_opts is not None:
                kw["compression_opts"] = compression_opts
        if chunks is not None:
            kw["chunks"] = chunks

        self._group.create_dataset(name, data=data, **kw)

    def write_attr(self, name: str, value) -> None:
        self._group.attrs[name] = value

    # ── read operations ───────────────────────────────────────────────

    def read_array(self, name: str) -> np.ndarray:
        ds = self._group[name]
        data = ds[()]
        if isinstance(data, bytes):
            return np.array(data.decode("utf-8"))
        if hasattr(data, "dtype") and data.dtype.kind == "O":
            decoded = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in np.asarray(data).flat
            ]
            return np.array(decoded).reshape(data.shape)
        return np.asarray(data)

    def read_attr(self, name: str):
        val = self._group.attrs[name]
        if isinstance(val, bytes):
            return val.decode("utf-8")
        return val

    def has_attr(self, name: str) -> bool:
        return name in self._group.attrs

    # ── lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()
