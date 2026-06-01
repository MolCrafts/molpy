# Changelog

## Unreleased

### Breaking changes

- **molrs is now a required dependency.** `molcrafts-molrs` moved from an
  optional extra into the core `dependencies`. The `molpy[molrs]` extra key was
  removed — installing molpy always installs molrs. `Frame`, `Block`, and `Box`
  are backed by (and inherit from) molrs types, and the `compute` operators
  forward directly into the Rust kernel. There is no pure-Python fallback.
- **The RDKit-backed compute node was removed.** `molpy.compute.rdkit`
  (`Generate3D` / `OptimizeGeometry` over `RDKitAdapter`) is gone.
  `molpy.compute.Generate3D` is now the molrs-backed trunk operator, taking an
  `Atomistic` graph and returning a fresh 3D structure. The RDKit adapter
  (`molpy.adapter.rdkit`) is retained as an
  **optional** external backend; `rdkit` remains an optional extra, not a
  required dependency.

### Added

- `molpy.compute.NeighborList` — linked-cell neighbor search (molrs backend).
- `molpy.compute.RDF` — radial distribution function over one or more frames.
- `molpy.Box` inherits `molrs.Box`, so a molpy box is accepted by any molrs API
  with no conversion.
- molrs analyses exposed as molpy operators: `MSD`, `Cluster`, `ClusterCenters`,
  `CenterOfMass`, `GyrationTensor`, `InertiaTensor`, `RadiusOfGyration`, `Pca`,
  `KMeans`.

### Changed

- `compute.mcd` and `compute.pmsd` now compute minimum-image displacements via
  molrs `Box.delta(minimum_image=True)`; public signatures are unchanged.

### Migration

- Replace `pip install "molcrafts-molpy[molrs]"` with `pip install molcrafts-molpy`.
- If you imported `from molpy.compute.rdkit import Generate3D`, switch to
  `from molpy.compute import Generate3D` (molrs-backed, `Atomistic -> Atomistic`)
  or, for the RDKit adapter flow, `from molpy.adapter import Generate3D`.

See [the molrs backend developer guide](developer/molrs-backend.md) for details.
