---
title: Reparent HDF5 reader onto pure BaseTrajectoryReader + fix exports
status: approved
created: 2026-05-31
---

# Reparent HDF5 reader onto pure BaseTrajectoryReader + fix exports

## Summary
With the pure `BaseTrajectoryReader` interface and `MmapTrajectoryReader`
landed (frame-reader-hierarchy-01-split), reparent `HDF5TrajectoryReader` —
which today subclasses nothing and re-implements the whole iteration API from
scratch — onto the pure `BaseTrajectoryReader`, deleting its duplicated
`read_frames`/`read_range`/`read_all`/`__iter__`/`__getitem__`/`__len__` so it
inherits them, keeping only its h5-specific `read_frame` + `n_frames`. Then fix
the export surfaces in `io/trajectory/__init__.py` and `io/__init__.py` so
`BaseReader`, `BaseTrajectoryReader`, and `MmapTrajectoryReader` are importable
(the downstream molexp feature keys on `molpy.io.BaseTrajectoryReader`), and
`FrameLocation` is exported from its new mmap location only. Outcome: the
interface is no longer duplicated; a binary random-access reader and the text
mmap readers share one pure `Iterable[Frame]` contract. Zero backward
compatibility — `FrameLocation` is not re-exported from its old path.

## Design
Builds strictly on 01 (cannot be merged before it):

- `HDF5TrajectoryReader(BaseTrajectoryReader)` — change `bases=[]` to subclass
  the pure `BaseTrajectoryReader`. DELETE its `read_frames` (if present),
  `read_range`-equivalent slicing, `__iter__`, `__getitem__`, and `__len__`;
  these are inherited from the pure base and defined purely in terms of
  `read_frame` + `n_frames`. KEEP its h5-specific `read_frame`,
  `_read_frame_from_file`, `n_frames` property, `_get_n_frames`,
  `_get_n_frames_from_file`. The base's concrete `__iter__` iterates
  `range(n_frames)` calling `read_frame`; HDF5's existing lazy `__iter__`
  (which opens the file once and yields sorted frame groups) is removed in
  favor of the inherited one — behavior is equivalent for the contiguous
  0..n-1 frame layout the writer produces. `__init__` continues to do its own
  h5py-specific path handling; it does not need `BaseReader.__init__` because
  HDF5 validates/normalizes its single path itself (it predates BaseReader and
  its `_path` attribute is h5-internal) — but it MUST satisfy the
  `BaseTrajectoryReader` abstract contract (`read_frame`, `n_frames`).
  Note: do not call `BaseReader.__init__` unless HDF5's `_path`/lifecycle is
  also migrated; keeping HDF5's own `__enter__`/`__exit__` is acceptable since
  the pure base's lifecycle comes from `BaseReader` only, and HDF5 overrides it.
- `io/trajectory/__init__.py` — update imports + `__all__`: import
  `BaseTrajectoryReader`, `MmapTrajectoryReader`, `FrameLocation`,
  `TrajectoryWriter` from `.base`; import `BaseReader` from `..base`; export
  all of them plus the concrete readers.
- `io/__init__.py` — update the `from .trajectory.base import (...)` block to
  add `MmapTrajectoryReader`; add `from .base import BaseReader`; add
  `BaseReader` and `MmapTrajectoryReader` to `__all__`; keep `FrameLocation`
  and `BaseTrajectoryReader` exported. Verify `from molpy.io import BaseReader,
  BaseTrajectoryReader, MmapTrajectoryReader, FrameLocation` all resolve.
- No back-compat re-export: there is no alias for `FrameLocation` at any old
  location, and no `BaseTrajectoryReader`-as-mmap shim.

## Files to create or modify
- src/molpy/io/trajectory/h5.py — `HDF5TrajectoryReader(BaseTrajectoryReader)`; delete duplicated `__iter__`/`__getitem__`/`__len__` (and any read_frames/read_range/read_all); keep h5-specific `read_frame`/`n_frames`/`_get_n_frames`/`_get_n_frames_from_file`/`_read_frame_from_file`.
- src/molpy/io/trajectory/__init__.py — update imports + `__all__`: add `BaseReader`, `MmapTrajectoryReader`; keep `BaseTrajectoryReader`, `FrameLocation`, `TrajectoryWriter`, concrete readers.
- src/molpy/io/__init__.py — add `from .base import BaseReader`; add `MmapTrajectoryReader` to the trajectory import block; add `BaseReader` + `MmapTrajectoryReader` to `__all__`.
- tests/test_io/test_trajectory/test_reader_hierarchy.py (new) — structural test groups (b) isinstance/inheritance and (c) parity.

## Tasks
- [ ] Write failing tests for the HDF5 reparent and import surface (tests/test_io/test_trajectory/test_reader_hierarchy.py): assert `isinstance(HDF5TrajectoryReader(...), BaseTrajectoryReader)` and `not isinstance(..., MmapTrajectoryReader)`; assert XYZ/LAMMPS readers are `isinstance` `MmapTrajectoryReader`; assert `from molpy.io import BaseReader, BaseTrajectoryReader, MmapTrajectoryReader, FrameLocation` all import
- [ ] Write failing parity tests (same file): on an h5 fixture trajectory, the reparented HDF5 reader yields identical frame count and identical first/last frame data via inherited `__iter__`/`__getitem__`/slice as before; on an XYZ and a LAMMPS fixture, frame count + first/last frame atom data are unchanged
- [ ] Reparent `HDF5TrajectoryReader` to `BaseTrajectoryReader` in src/molpy/io/trajectory/h5.py and delete its duplicated iteration methods, keeping h5-specific `read_frame`+`n_frames`
- [ ] Update src/molpy/io/trajectory/__init__.py imports and `__all__` to export `BaseReader`, `MmapTrajectoryReader`, `BaseTrajectoryReader`, `FrameLocation`
- [ ] Update src/molpy/io/__init__.py to import/export `BaseReader` and `MmapTrajectoryReader` and verify `from molpy.io import BaseReader, BaseTrajectoryReader` resolves
- [ ] Run full check + test suite (ruff check src tests && ty check src/molpy/ && pytest tests/ -m "not external" -v)

## Testing strategy
- New structural test group (b): `isinstance(HDF5TrajectoryReader(fixture),
  BaseTrajectoryReader)` is True and `isinstance(..., MmapTrajectoryReader)` is
  False; `isinstance(XYZTrajectoryReader(fixture), MmapTrajectoryReader)` and
  `isinstance(LammpsTrajectoryReader(fixture), MmapTrajectoryReader)` are True.
- New structural test group (c): parity — for an XYZ and a LAMMPS fixture
  trajectory, the reparented readers produce identical frame count and
  identical first/last frame atom data (element/coordinates for XYZ; atom rows
  + box for LAMMPS) compared to the pre-refactor expectation encoded in the
  existing fixtures; for the h5 fixture, frame count and first/last frame data
  via the now-inherited `__iter__`/`__getitem__`/slice match the prior values.
- Import surface: `from molpy.io import BaseReader, BaseTrajectoryReader,
  MmapTrajectoryReader, FrameLocation` and `from molpy.io.trajectory import
  BaseReader, MmapTrajectoryReader` all succeed; `__all__` contains the new
  names.
- Regression gate: full existing suite incl. the h5/lammps trajectory tests,
  mmap lifecycle, prefetch, index, and tests/test_core/test_trajectory.py pass
  unchanged.

## Out of scope
- The `BaseReader` / pure-vs-mmap split itself — landed in
  frame-reader-hierarchy-01-split.
- Migrating HDF5's `_path`/lifecycle onto `BaseReader.__init__` — HDF5 keeps
  its own h5py path handling; only the `BaseTrajectoryReader` abstract contract
  is required.
- Any new reader format; the molexp discovery/QM9 reference subclass; molvis.
- Rewriting the stale docs prose in docs/tutorials/io.md and
  docs/developer/extending-io.md (pre-existing drift).
