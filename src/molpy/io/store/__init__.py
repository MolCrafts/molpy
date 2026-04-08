"""Unified molecular simulation store.

Supports both HDF5 (.h5 / .hdf5) and Zarr (.zarr) backends.
Both h5py and zarr are optional dependencies — only the backend you use
needs to be installed.

Store layout::

    /
    meta/              flexible metadata (units, provenance, ...)
    frame/             static system (atoms, bonds, simbox)
    forcefield/        interaction parameters
    trajectory/        time-dependent arrays
      step, time       trajectory time index
      x, y, z          trajectory coordinates
      box_h            trajectory box
      pe, ke, ...      trajectory scalars

Usage::

    from molpy.io.store import MolStore

    # Write
    with MolStore("sim.zarr", mode="w") as store:
        store.write_meta(units_style="real")
        store.write_frame(frame)
        store.write_forcefield(ff)
        store.write_trajectory(frames)

    # Read
    with MolStore("sim.zarr", mode="r") as store:
        frame = store.read_frame()
        ff    = store.read_forcefield()
        for f in store.iter_trajectory():
            ...
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    from molpy.core import Box, Frame
    from molpy.core.forcefield import AtomisticForcefield, ForceField

PathLike = str | Path

# ─────────────────────────────────────────────────────────────────────
# Backend dispatch
# ─────────────────────────────────────────────────────────────────────

_H5_EXTENSIONS = {".h5", ".hdf5"}
_ZARR_EXTENSIONS = {".zarr"}


def _open_backend(path: PathLike, mode: str):
    ext = Path(path).suffix.lower()
    if ext in _H5_EXTENSIONS:
        from ._h5 import H5Backend

        return H5Backend(str(path), mode)
    elif ext in _ZARR_EXTENSIONS:
        from ._zarr import ZarrBackend

        return ZarrBackend(str(path), mode)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Use .h5/.hdf5 for HDF5 or .zarr for Zarr."
        )


# ─────────────────────────────────────────────────────────────────────
# Trajectory array names (stored under trajectory/ group)
# ─────────────────────────────────────────────────────────────────────

_TRAJ_PERATOM_KEYS = {"x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"}
_TRAJ_SCALAR_KEYS = {"temp", "press", "pe", "ke", "etotal"}

# Default chunk sizes
_SCALAR_CHUNK = 100
_BOX_H_CHUNK_T = 1


# ─────────────────────────────────────────────────────────────────────
# MolStore
# ─────────────────────────────────────────────────────────────────────


class MolStore:
    """Read/write molecular simulation data.

    Store layout::

        /
        meta/              flexible metadata (units, provenance, ...)
        frame/             reference frame
          atoms/           per-atom arrays [N]
          simbox/          h [3,3], origin [3], pbc [3]
          bonds/           i, j [B] (UInt32)
        trajectory/        time-dependent arrays
          step [F]         Int64, trajectory step numbers
          time [F]         Float64, simulation time
          x, y, z [F, N]  Float32, trajectory coordinates
          box_h [F, 3, 3] Float32, trajectory box
          pe, ke, ... [F]  Float64, trajectory scalars
        forcefield/        interaction parameters
    """

    FORMAT_NAME = "molpy-zarr"

    def __init__(self, path: PathLike, mode: str = "r"):
        self._path = Path(path)
        self._mode = mode
        self._backend = _open_backend(path, mode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def _get_backend(self):
        """Return the backend, raising if the store has been closed."""
        if self._backend is None:
            raise RuntimeError("Store has been closed")
        return self._backend

    # ─────────────────────────────────────────────────────────────────
    # meta/
    # ─────────────────────────────────────────────────────────────────

    def write_meta(self, **kwargs: Any) -> None:
        """Write metadata to the meta/ group.

        The meta/ group is a flexible container for arbitrary key-value
        pairs.  Defaults for ``format_name``, ``program``, and
        ``created_utc`` are auto-populated if not provided.

        Examples::

            store.write_meta(units_style="real")
            store.write_meta(units_style="si", description="NVT run")
        """
        _ver: str
        try:
            from molpy.version import version as _v

            _ver = _v
        except Exception:
            _ver = "unknown"

        defaults = {
            "format_name": self.FORMAT_NAME,
            "program": f"molpy {_ver}",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        # User-supplied kwargs override defaults
        merged = {**defaults, **kwargs}

        meta = self._get_backend().create_group("meta")
        for key, value in merged.items():
            if isinstance(value, dict):
                # Nested group
                sg = meta.create_group(key)
                for sk, sv in value.items():
                    sg.write_attr(sk, sv)
            else:
                meta.write_attr(key, value)

    def read_meta(self) -> dict[str, Any]:
        """Read all metadata from the meta/ group.

        Returns a dict of all key-value pairs found as attributes,
        plus nested dicts for any sub-groups.
        """
        backend = self._get_backend()
        if "meta" not in backend.list_groups():
            return {}
        meta = backend["meta"]
        result: dict[str, Any] = {}
        # Read all attributes
        for key in meta.list_members():
            if meta.has_attr(key):
                result[key] = meta.read_attr(key)
        # Also read attrs that are not member names
        for key in ("format_name", "program", "created_utc", "units_style"):
            if key not in result and meta.has_attr(key):
                result[key] = meta.read_attr(key)
        # Read sub-groups
        for gname in meta.list_groups():
            sg = meta[gname]
            sub = {}
            for sk in sg.list_members():
                if sg.has_attr(sk):
                    sub[sk] = sg.read_attr(sk)
            result[gname] = sub
        return result

    # ─────────────────────────────────────────────────────────────────
    # frame/
    # ─────────────────────────────────────────────────────────────────

    def write_frame(self, frame: "Frame") -> None:
        fg = self._get_backend().create_group("frame")

        # ── atoms ────────────────────────────────────────────────────
        if "atoms" in frame:
            ag = fg.create_group("atoms")
            for key, arr in frame["atoms"]._vars.items():
                data = np.asarray(arr)
                if key in ("x", "y", "z"):
                    data = data.astype(np.float32)
                ag.write_array(key, data)

        # ── topology blocks ──────────────────────────────────────────
        for topo_name in ("bonds", "angles", "dihedrals", "impropers"):
            if topo_name in frame:
                tg = fg.create_group(topo_name)
                for key, arr in frame[topo_name]._vars.items():
                    data = np.asarray(arr)
                    if key in ("atomi", "atomj", "atomk", "atoml"):
                        data = data.astype(np.uint32)
                    tg.write_array(key, data)

        # ── simbox ─────────────────────────────────────────────────────
        box = frame.box
        if box is not None:
            sb = fg.create_group("simbox")
            sb.write_array("h", np.asarray(box.matrix, dtype=np.float32))
            sb.write_array("origin", np.asarray(box.origin, dtype=np.float32))
            sb.write_array("pbc", np.asarray(box.pbc, dtype=np.uint8))

    def read_frame(self) -> "Frame":
        from molpy.core import Box, Frame
        from molpy.core.frame import Block

        fg = self._get_backend()["frame"]
        frame = Frame()

        # ── atoms ────────────────────────────────────────────────────
        if fg.has("atoms"):
            ag = fg["atoms"]
            atoms = Block()
            for key in ag.list_arrays():
                atoms[key] = ag.read_array(key)
            frame["atoms"] = atoms

        # ── topology blocks ──────────────────────────────────────────
        for topo_name in ("bonds", "angles", "dihedrals", "impropers"):
            if fg.has(topo_name):
                tg = fg[topo_name]
                block = Block()
                for key in tg.list_arrays():
                    block[key] = tg.read_array(key)
                frame[topo_name] = block

        # ── simbox ─────────────────────────────────────────────────────
        if fg.has("simbox"):
            sb = fg["simbox"]
            h = sb.read_array("h")
            origin = (
                sb.read_array("origin") if "origin" in sb.list_arrays() else np.zeros(3)
            )
            pbc = (
                sb.read_array("pbc")
                if "pbc" in sb.list_arrays()
                else np.ones(3, dtype=bool)
            )
            frame.box = Box(matrix=h, origin=origin, pbc=pbc.astype(bool))
        elif fg.has("box"):
            # Backward compat: old format stored box/matrix
            matrix = fg["box"].read_array("matrix")
            frame.box = Box(matrix=matrix)

        return frame

    # ─────────────────────────────────────────────────────────────────
    # forcefield/
    # ─────────────────────────────────────────────────────────────────

    def write_forcefield(self, ff: "ForceField") -> None:
        from molpy.core.forcefield import (
            AngleStyle,
            AngleType,
            AtomStyle,
            AtomType,
            BondStyle,
            BondType,
            DihedralStyle,
            DihedralType,
            ImproperStyle,
            ImproperType,
            PairStyle,
            PairType,
            Style,
        )

        ffg = self._get_backend().create_group("forcefield")
        ffg.write_attr("units_style_ref", ff.units)

        # ── build atom-type name→id mapping ──────────────────────────
        atom_types = ff.get_types(AtomType)
        atom_types.sort(key=lambda t: t.name)
        name_to_id: dict[str, int] = {}
        for i, at in enumerate(atom_types, start=1):
            name_to_id[at.name] = i

        if atom_types:
            ffg.write_array(
                "atom_type_names",
                np.array([at.name for at in atom_types]),
            )

        # ── style declarations + coeff tables ────────────────────────
        style_map = [
            ("pair", PairStyle, PairType),
            ("bond", BondStyle, BondType),
            ("angle", AngleStyle, AngleType),
            ("dihedral", DihedralStyle, DihedralType),
            ("improper", ImproperStyle, ImproperType),
        ]

        for label, style_cls, type_cls in style_map:
            styles = ff.get_styles(style_cls)
            if not styles:
                continue

            style = styles[0]
            style_str = style.name
            if style.params.args:
                style_str += " " + " ".join(str(a) for a in style.params.args)
            ffg.write_attr(f"{label}_style", style_str)

            types = style.get_types(type_cls)
            if not types:
                continue
            types = sorted(types, key=lambda t: t.name)

            coeff_group = ffg.create_group(f"{label}_coeff")
            self._write_coeff_table(coeff_group, label, types, name_to_id)

        # ── atom style ───────────────────────────────────────────────
        atom_styles = ff.get_styles(AtomStyle)
        if atom_styles:
            style_str = atom_styles[0].name
            if atom_styles[0].params.args:
                style_str += " " + " ".join(str(a) for a in atom_styles[0].params.args)
            ffg.write_attr("atom_style", style_str)

        # ── kspace_style ─────────────────────────────────────────────
        kspace = getattr(ff, "_kspace_style", None)
        if kspace:
            ffg.write_attr("kspace_style", kspace)

        # ── mixing_rule, special_bonds ───────────────────────────────
        mixing = getattr(ff, "_mixing_rule", None)
        if mixing:
            ffg.write_attr("mixing_rule", mixing)

        sb_lj = getattr(ff, "_special_bonds_lj", None)
        if sb_lj is not None:
            sb = ffg.create_group("special_bonds")
            sb.write_array("lj", np.asarray(sb_lj, dtype=np.float64))

        sb_coul = getattr(ff, "_special_bonds_coul", None)
        if sb_coul is not None:
            if not ffg.has("special_bonds"):
                sb = ffg.create_group("special_bonds")
            else:
                sb = ffg["special_bonds"]
            sb.write_array("coul", np.asarray(sb_coul, dtype=np.float64))

    @staticmethod
    def _write_coeff_table(group, label, types, name_to_id):
        """Write a normalized coefficient table for an interaction type."""
        if not types:
            return

        all_param_keys: list[str] = []
        seen: set[str] = set()
        for t in types:
            for k in t.params.kwargs:
                if k not in seen:
                    all_param_keys.append(k)
                    seen.add(k)

        M = len(types)
        P = len(all_param_keys)

        if label == "pair":
            ii = np.empty(M, dtype=np.int32)
            jj = np.empty(M, dtype=np.int32)
            for r, t in enumerate(types):
                ii[r] = name_to_id.get(t.itom.name, 0)
                jj[r] = name_to_id.get(t.jtom.name, 0)
            group.write_array("i", ii)
            group.write_array("j", jj)
        else:
            type_ids = np.arange(1, M + 1, dtype=np.int32)
            group.write_array("type", type_ids)

        params = np.zeros((M, P), dtype=np.float64)
        for r, t in enumerate(types):
            for c, k in enumerate(all_param_keys):
                val = t.params.kwargs.get(k)
                if val is not None:
                    params[r, c] = float(val)

        group.write_array("params", params)
        group.write_array("param_names", np.array(all_param_keys))

    def read_forcefield(self) -> "AtomisticForcefield":
        from molpy.core.forcefield import AtomisticForcefield, AtomType

        ffg = self._get_backend()["forcefield"]
        ff = AtomisticForcefield()

        if ffg.has_attr("units_style_ref"):
            ff.units = ffg.read_attr("units_style_ref")

        # ── restore atom type name→id mapping ────────────────────────
        id_to_name: dict[int, str] = {}
        atom_type_objs: dict[str, AtomType] = {}

        if "atom_type_names" in ffg.list_arrays():
            names = ffg.read_array("atom_type_names")
            for i, name in enumerate(names, start=1):
                id_to_name[i] = str(name)

        atom_style_str = ""
        if ffg.has_attr("atom_style"):
            atom_style_str = ffg.read_attr("atom_style")
        parts = atom_style_str.split() if atom_style_str else ["full"]
        atom_style = ff.def_atomstyle(parts[0])
        for tid, name in id_to_name.items():
            at = atom_style.def_type(name)
            atom_type_objs[name] = at

        _placeholder_at = atom_style.def_type("_placeholder")

        def _get_atom_type(tid: int) -> AtomType:
            name = id_to_name.get(tid)
            if name is None:
                return _placeholder_at
            if name not in atom_type_objs:
                atom_type_objs[name] = atom_style.def_type(name)
            return atom_type_objs[name]

        # ── read styles and coefficient tables ───────────────────────
        style_defs = [
            ("pair", "def_pairstyle", self._read_pair_coeff),
            ("bond", "def_bondstyle", self._read_bonded_coeff),
            ("angle", "def_anglestyle", self._read_bonded_coeff),
            ("dihedral", "def_dihedralstyle", self._read_bonded_coeff),
            ("improper", "def_improperstyle", self._read_bonded_coeff),
        ]

        for label, def_method, reader_fn in style_defs:
            style_attr = f"{label}_style"
            if not ffg.has_attr(style_attr):
                continue

            style_str = ffg.read_attr(style_attr)
            parts = style_str.split()
            style_name = parts[0]
            style_args = []
            for a in parts[1:]:
                try:
                    style_args.append(float(a))
                except ValueError:
                    style_args.append(a)

            style = getattr(ff, def_method)(style_name, *style_args)

            coeff_name = f"{label}_coeff"
            if ffg.has(coeff_name):
                cg = ffg[coeff_name]
                reader_fn(cg, style, label, _get_atom_type)

        return ff

    @staticmethod
    def _read_pair_coeff(cg, style, label, get_atom_type):
        """Read pair coefficient table and populate style with PairTypes."""
        param_names = cg.read_array("param_names")
        params = cg.read_array("params")
        ii = cg.read_array("i")
        jj = cg.read_array("j")

        for r in range(len(ii)):
            at_i = get_atom_type(int(ii[r]))
            at_j = get_atom_type(int(jj[r]))
            kwargs = {
                str(param_names[c]): float(params[r, c]) for c in range(params.shape[1])
            }
            style.def_type(at_i, at_j, **kwargs)

    @staticmethod
    def _read_bonded_coeff(cg, style, label, get_atom_type):
        """Read bond/angle/dihedral/improper coefficient table."""
        from molpy.core.forcefield import (
            AngleType,
            BondType,
            DihedralType,
            ImproperType,
        )

        param_names = cg.read_array("param_names")
        params = cg.read_array("params")
        type_ids = cg.read_array("type")

        ph = get_atom_type(-1)  # returns _placeholder

        type_cls_map = {
            "bond": (BondType, lambda name, **kw: BondType(name, ph, ph, **kw)),
            "angle": (AngleType, lambda name, **kw: AngleType(name, ph, ph, ph, **kw)),
            "dihedral": (
                DihedralType,
                lambda name, **kw: DihedralType(name, ph, ph, ph, ph, **kw),
            ),
            "improper": (
                ImproperType,
                lambda name, **kw: ImproperType(name, ph, ph, ph, ph, **kw),
            ),
        }

        _, factory = type_cls_map[label]

        for r in range(len(type_ids)):
            kwargs = {
                str(param_names[c]): float(params[r, c]) for c in range(params.shape[1])
            }
            type_name = f"{label}_type_{int(type_ids[r])}"
            t = factory(type_name, **kwargs)
            style.types.add(t)

    # ─────────────────────────────────────────────────────────────────
    # trajectory — stored under trajectory/ group
    # ─────────────────────────────────────────────────────────────────

    def write_trajectory(
        self,
        frames: list["Frame"],
        timesteps: list[int] | np.ndarray | None = None,
        times: list[float] | np.ndarray | None = None,
    ) -> None:
        """Write an entire trajectory at once (batch mode).

        Arrays are stored under the trajectory/ group:
        trajectory/step [F], trajectory/time [F],
        trajectory/x [F,N], trajectory/y [F,N], trajectory/z [F,N],
        trajectory/box_h [F,3,3], trajectory/pe [F], etc.
        """
        if not frames:
            return

        T = len(frames)
        tg = self._get_backend().create_group("trajectory")

        # ── step ─────────────────────────────────────────────────────
        if timesteps is None:
            step_arr = np.arange(T, dtype=np.int64)
        else:
            step_arr = np.asarray(timesteps, dtype=np.int64)
        tg.write_array("step", step_arr, chunks=(min(T, _SCALAR_CHUNK),))

        # ── time (optional) ──────────────────────────────────────────
        if times is not None:
            time_arr = np.asarray(times, dtype=np.float64)
            tg.write_array("time", time_arr, chunks=(min(T, _SCALAR_CHUNK),))

        # ── per-atom arrays under trajectory/ ────────────────────────
        first = frames[0]
        if "atoms" not in first:
            return

        atom_keys = list(first["atoms"]._vars.keys())
        N = first["atoms"].nrows

        keys_to_store = [k for k in atom_keys if k in _TRAJ_PERATOM_KEYS]

        for key in keys_to_store:
            stacked = np.empty((T, N), dtype=np.float32)
            for t, f in enumerate(frames):
                stacked[t, :] = f["atoms"][key]
            # Chunk [1, N] — one frame per chunk
            tg.write_array(key, stacked, chunks=(1, N))

        # ── box_h under trajectory/ ──────────────────────────────────
        has_box = any("box" in f.metadata for f in frames)
        if has_box:
            box_data = np.empty((T, 3, 3), dtype=np.float32)
            for t, f in enumerate(frames):
                box = f.box
                if box is not None:
                    box_data[t] = np.asarray(box)
                else:
                    box_data[t] = 0.0
            # Chunk [1, 3, 3] — one frame per chunk
            tg.write_array("box_h", box_data, chunks=(_BOX_H_CHUNK_T, 3, 3))

        # ── scalar observables under trajectory/ ─────────────────────
        for key in _TRAJ_SCALAR_KEYS:
            if key not in frames[0].metadata:
                continue
            arr = np.array(
                [f.metadata.get(key, np.nan) for f in frames],
                dtype=np.float64,
            )
            tg.write_array(key, arr, chunks=(min(T, _SCALAR_CHUNK),))

    @property
    def n_trajectory_frames(self) -> int:
        backend = self._get_backend()
        if "trajectory" not in backend.list_groups():
            return 0
        tg = backend["trajectory"]
        if "step" not in tg.list_arrays():
            return 0
        return len(tg.read_array("step"))

    def read_trajectory_frame(self, index: int) -> "Frame":
        """Read a single trajectory frame by index."""
        from molpy.core import Box, Frame
        from molpy.core.frame import Block

        T = self.n_trajectory_frames

        if index < 0:
            index = T + index
        if index < 0 or index >= T:
            raise IndexError(f"Trajectory index {index} out of range [0, {T})")

        tg = self._get_backend()["trajectory"]
        frame = Frame()

        # ── step/time metadata ───────────────────────────────────────
        step_arr = tg.read_array("step")
        frame.metadata["timestep"] = int(step_arr[index])

        if "time" in tg.list_arrays():
            time_arr = tg.read_array("time")
            frame.metadata["time"] = float(time_arr[index])

        # ── per-atom data from trajectory/ arrays ────────────────────
        traj_arrays = set(tg.list_arrays())
        peratom_present = [k for k in _TRAJ_PERATOM_KEYS if k in traj_arrays]

        if peratom_present:
            atoms = Block()
            for key in peratom_present:
                arr = tg.read_array(key)
                atoms[key] = arr[index]
            frame["atoms"] = atoms

        # ── box_h ────────────────────────────────────────────────────
        if "box_h" in traj_arrays:
            box_arr = tg.read_array("box_h")
            frame.box = Box(matrix=box_arr[index])

        # ── scalar observables ───────────────────────────────────────
        for key in _TRAJ_SCALAR_KEYS:
            if key in traj_arrays:
                arr = tg.read_array(key)
                frame.metadata[key] = float(arr[index])

        return frame

    def iter_trajectory(self) -> Iterator["Frame"]:
        """Lazily iterate over all trajectory frames."""
        T = self.n_trajectory_frames
        for i in range(T):
            yield self.read_trajectory_frame(i)

    def read_trajectory_arrays(self) -> dict[str, np.ndarray]:
        """Read raw trajectory arrays without creating Frame objects.

        Returns a dict like::

            {
                "step": [T] int64,
                "x": [T, N] float32,
                "y": [T, N] float32,
                ...
                "box_h": [T, 3, 3] float32,
                "pe": [T] float64,
                ...
            }
        """
        result: dict[str, np.ndarray] = {}

        backend = self._get_backend()
        if "trajectory" not in backend.list_groups():
            return result

        tg = backend["trajectory"]
        traj_arrays = set(tg.list_arrays())

        # step/time
        for key in ("step", "time"):
            if key in traj_arrays:
                result[key] = tg.read_array(key)

        # per-atom
        for key in _TRAJ_PERATOM_KEYS:
            if key in traj_arrays:
                result[key] = tg.read_array(key)

        # box_h
        if "box_h" in traj_arrays:
            result["box_h"] = tg.read_array("box_h")

        # scalars
        for key in _TRAJ_SCALAR_KEYS:
            if key in traj_arrays:
                result[key] = tg.read_array(key)

        return result


# ─────────────────────────────────────────────────────────────────────
# Factory functions
# ─────────────────────────────────────────────────────────────────────


def open_store(path: PathLike, mode: str = "r") -> MolStore:
    """Open a molecular simulation store.

    Args:
        path: Path to .h5/.hdf5 or .zarr store.
        mode: 'r' for read, 'w' for write, 'a' for append.

    Returns:
        MolStore instance.
    """
    return MolStore(path, mode=mode)
