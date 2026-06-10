# Changelog

The full user-facing changelog lives in [docs/changelog.md](docs/changelog.md).
This file records API renames and breaking changes at the repository root.

## Unreleased

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
