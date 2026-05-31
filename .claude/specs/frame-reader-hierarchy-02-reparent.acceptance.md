---
slug: frame-reader-hierarchy-02-reparent
criteria:
  - id: ac-001
    summary: HDF5TrajectoryReader is a BaseTrajectoryReader, not a MmapTrajectoryReader
    type: code
    pass_when: |
      For an h5 fixture, isinstance(HDF5TrajectoryReader(path),
      BaseTrajectoryReader) is True and isinstance(..., MmapTrajectoryReader)
      is False; asserted under pytest in
      tests/test_io/test_trajectory/test_reader_hierarchy.py.
    status: pending
  - id: ac-002
    summary: HDF5 reader inherits iteration API, defines only read_frame+n_frames
    type: code
    pass_when: |
      The HDF5TrajectoryReader class body no longer defines __iter__,
      __getitem__, or __len__ (they resolve to BaseTrajectoryReader); it still
      defines read_frame and the n_frames property; and iterating the h5
      fixture yields n_frames frames with first/last frame data unchanged.
    status: pending
  - id: ac-003
    summary: XYZ/LAMMPS readers are MmapTrajectoryReader instances
    type: code
    pass_when: |
      isinstance(XYZTrajectoryReader(xyz_fixture), MmapTrajectoryReader) and
      isinstance(LammpsTrajectoryReader(lammps_fixture), MmapTrajectoryReader)
      are both True under pytest.
    status: pending
  - id: ac-004
    summary: XYZ/LAMMPS/HDF5 parse fixtures with unchanged count + first/last frame
    type: code
    pass_when: |
      For the XYZ, LAMMPS, and HDF5 fixtures, len(reader) equals the expected
      frame count and reader[0] / reader[-1] expose the same atom data
      (XYZ: element + x/y/z; LAMMPS: atom rows + box; HDF5: block columns) as
      the existing fixture expectations; asserted under pytest.
    status: pending
  - id: ac-005
    summary: BaseReader/BaseTrajectoryReader/MmapTrajectoryReader/FrameLocation importable from molpy.io
    type: code
    pass_when: |
      `from molpy.io import BaseReader, BaseTrajectoryReader,
      MmapTrajectoryReader, FrameLocation` succeeds, and all four names are in
      molpy.io.__all__; `from molpy.io.trajectory import BaseReader,
      MmapTrajectoryReader, BaseTrajectoryReader, FrameLocation` also succeeds.
    status: pending
  - id: ac-006
    summary: Full check + test gate passes
    type: runtime
    pass_when: |
      `ruff check src tests && ty check src/molpy/` exits 0 and
      `pytest tests/ -m "not external" -v` exits 0.
    status: pending
---

# Acceptance criteria

- ac-001..ac-002 prove the HDF5 reader is now a pure-base subclass with the
  duplicated iteration API removed (test group (b) plus inheritance proof).
- ac-003..ac-004 prove the concrete readers' family membership and
  behavior-parity on fixtures (test groups (b)/(c)).
- ac-005 fixes the import/export surface the downstream molexp feature depends
  on.
- ac-006 is the project-wide quality gate.
