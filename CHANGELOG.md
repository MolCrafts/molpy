# Changelog

The full user-facing changelog lives in [docs/changelog.md](docs/changelog.md).
This file records API renames and breaking changes at the repository root.

## 0.4.2 - 2026-06-18

### Added (additive; no API renames or breaking changes)

- GROMACS **TRR** / **XTC** and **DCD** trajectory readers
  (`read_trr_trajectory`, `read_xtc_trajectory`, `read_dcd_trajectory`) and TRR/
  XTC writers (`write_trr`, `write_xtc`), thin delegations to the molrs backend.

### Changed

- Pin `molcrafts-molrs==0.1.4` (was `0.1.3`) — required for the new GROMACS
  trajectory bindings.

## 0.4.0 - 2026-06-11

### Removed (no deprecation shims — project is in the experimental stage)

- `molpy.Topology` / `molpy.core.Topology` (the parallel igraph engine was
  deleted); `Atomistic.get_topo()` now always returns an `Atomistic`.
- Python potential kernels under `molpy.potential` (now a facade over molrs
  Styles); `potential.base.Potential`; `optimize.potential_wrappers` →
  `ForceFieldPotential`. Parameter `k0` → `k`; `def_type` params keyword-only.
- `Compute.execute()` / `input_key` / `output_key` / `_compute` hook /
  `Compute[InT, OutT]` generic — `Compute` is a plain `__init__` + `__call__`
  class.
- molpy's own LAMMPS/XYZ trajectory readers and mmap-index infrastructure
  (`Trajectory` is a `molrs.Trajectory` subclass; parsing lives in molrs).
- The `molpy.op` package (dead geometry helpers).
- Typifier surface: `ForceFieldAtomisticTypifier` → `ForceFieldTypifier`;
  `Opls{Bond,Angle,Dihedral}Typifier` and the public atom-only typifiers
  removed (`_OplsAtomTypifier` / `_GaffAtomTypifier` are private);
  `GaffAtomisticTypifier` alias dropped — use `OplsTypifier` / `GaffTypifier`.

### Behavior changes (fail-fast)

- Selectors raise `KeyError` on a missing column instead of matching nothing.
- The typifier raises on malformed SMARTS / invalid element instead of
  silently skipping or wildcarding.
- `ForceField.to_potentials()` raises on broken styles instead of dropping them.
- The LAMMPS data reader retains force-field coeffs and raises `ValueError`
  on malformed coeff lines.

### Renames (no deprecation shims — project is in the experimental stage)

- `ReactionPresetSpec.site_selector_left` / `site_selector_right` →
  `anchor_selector_left` / `anchor_selector_right`. The silent remapping
  inside `ReactionPresets.get()` is gone; specs now use the same anchor
  terminology as `Reacter` itself.
- `molpy.reacter.find_port_atom` → `molpy.reacter.find_port` (the old
  alias pair collapsed to the short name; `find_port_atom_by_node` is
  unchanged and `find_port(..., node_id=...)` covers the same use case).

### Surface consolidation

- Agent-only Tool classes (`PrepareMonomer`, `BuildPolymer`, `PlanSystem`,
  `BuildSystem`, `BuildPolymerAmber`) moved to
  `molpy.builder.polymer.tools` and are no longer exported from
  `molpy.builder` / `molpy.builder.polymer`.
- `ReactionPresets` / `ReactionPresetSpec` are now public exports of
  `molpy.builder.polymer` (the extension point for custom chemistry).
- `GBigSmilesCompiler`, `SystemPlanner`, and `PolydisperseChainGenerator`
  left the public `__all__` (still importable as internals).
- `molpy.builder`, `molpy.reacter`, `molpy.pack`, and `molpy.compute` are
  reachable as lazy top-level attributes (`mp.builder.…`).
