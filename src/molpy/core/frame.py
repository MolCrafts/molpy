import numpy as np
import xarray as xr
from xarray import DataTree
from collections.abc import MutableMapping
from typing import Any, Sequence

from .box import Box


def _dict_to_dataset(data: dict) -> xr.Dataset:
    """Convert a mapping of arrays to an ``xarray.Dataset`` with a common
    ``index`` dimension."""
    variables = {}
    for name, value in data.items():
        arr = np.asarray(value)
        dims = ["index"] + [f"dim_{i}" for i in range(1, arr.ndim)]
        variables[name] = (dims, arr)
    return xr.Dataset(variables)


class Frame(MutableMapping):
    """Container of simulation data based on :class:`xarray.Dataset` and
    :class:`datatree.DataTree`.``"""

    box: Box | None = None

    def __new__(cls, data: dict[str, Any] | None = None, *, style: str = "atomic"):
        if cls is Frame and style == "atomic":
            return super().__new__(AllAtomFrame)
        return super().__new__(cls)

    def __init__(self, data: dict[str, Any] | None = None, *_, **__):
        self._tree = DataTree(name="root")
        self.box = None
        self._scalars: dict[str, Any] = {}
        if data:
            for key, value in data.items():
                self[key] = value

    # mapping protocol -----------------------------------------------------
    def __getitem__(self, key: str):
        if key == "box":
            return self.box
        if key in self._tree:
            return self._tree[key].ds
        return self._scalars[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "box":
            self.box = value
            return
        if isinstance(value, xr.Dataset):
            ds = value
        elif isinstance(value, dict):
            ds = _dict_to_dataset(value)
        elif np.isscalar(value):
            self._scalars[key] = value
            return
        else:
            raise TypeError("Frame values must be xarray.Dataset or mapping")
        self._tree[key] = DataTree(ds, name=key)

    def __delitem__(self, key: str) -> None:
        if key == "box":
            self.box = None
        elif key in self._tree:
            del self._tree[key]
        else:
            del self._scalars[key]

    def __iter__(self):
        for k in self._tree.keys():
            yield k
        for k in self._scalars.keys():
            yield k
        if self.box is not None:
            yield "box"

    def __len__(self) -> int:  # number of datasets + box if present
        n = len(self._tree) + len(self._scalars)
        if self.box is not None:
            n += 1
        return n

    def __repr__(self) -> str:
        return f"<Frame: {list(self._tree.keys())}>"

    # convenience methods -------------------------------------------------
    def copy(self) -> "Frame":
        data = {k: self[k].copy() for k in self._tree.keys()}
        if self.box is not None:
            data["box"] = self.box
        return self.__class__(data)

    @classmethod
    def from_frames(cls, others: Sequence["Frame"]) -> "Frame":
        frame = cls()
        keys = set().union(*(f._tree.keys() for f in others))
        for key in keys:
            datasets = [f[key] for f in others if key in f._tree]
            frame[key] = xr.concat(datasets, dim="index")
        if any(f.box is not None for f in others):
            frame.box = others[0].box
        return frame

    def concat(self, other: "Frame") -> "Frame":
        return self.from_frames([self, other])

    def split(self, masks: Sequence[int] | Sequence[bool] | np.ndarray) -> list["Frame"]:
        masks = np.asarray(masks)
        if masks.dtype == bool:
            groups = [masks]
        else:
            groups = [masks == i for i in np.unique(masks)]
        frames = []
        for mask in groups:
            f = self.__class__()
            for key in self._tree.keys():
                ds = self[key]
                if "index" in ds.dims:
                    f[key] = ds.sel(index=mask)
                else:
                    f[key] = ds.copy()
            if self.box is not None:
                f.box = self.box
            frames.append(f)
        return frames

    def to_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """Return a nested ``dict`` representation of the frame."""
        data: dict[str, Any] = {k: {n: v.values for n, v in self[k].data_vars.items()} for k in self._tree.keys()}
        data.update(self._scalars)
        if self.box is not None:
            data["box"] = self.box.to_dict()
        return data

    def to_struct(self):
        from .struct import Entities, Struct
        import molpy as mp

        data = self.to_dict()
        struct = Struct()

        if "atoms" in data:
            atoms = data["atoms"]
            n_atoms = len(next(iter(atoms.values()), []))
            for i in range(n_atoms):
                struct.def_atom(**{k: np.asarray(v)[i] for k, v in atoms.items()})

        if "bonds" in data:
            struct["bonds"] = Entities()
            bonds = data["bonds"]
            n_bonds = len(next(iter(bonds.values()), []))
            for i in range(n_bonds):
                bond = {k: np.asarray(v)[i] for k, v in bonds.items()}
                i_id = int(bond.pop("i"))
                j_id = int(bond.pop("j"))
                itom = struct["atoms"].get_by(lambda a: a["id"] == i_id)
                jtom = struct["atoms"].get_by(lambda a: a["id"] == j_id)
                struct["bonds"].add(mp.Bond(itom, jtom, **bond))
        return struct

    def __add__(self, other: "Frame") -> "Frame":
        return self.concat(other)

    def __mul__(self, n: int) -> "Frame":
        return self.from_frames([self.copy() for _ in range(n)])


class AllAtomFrame(Frame):
    def __init__(self, data: dict[str, Any] | None = None, *_, **__):
        data = data or {}
        for key in ["atoms", "bonds", "angles", "dihedrals", "impropers"]:
            data.setdefault(key, {})
        super().__init__(data)
