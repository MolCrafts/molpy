# Performance notes

## Polymer build loop complexity

The polymer build loop (`PolymerBuilder._build_from_graph`) is designed so
that **per-connection bookkeeping is bounded by monomer size and live port
count, not by chain length**:

- **Reacter copy semantics** — `Reacter.run` copies its two inputs once
  each. With `record_intermediates=False`, the base `Reacter` never copies
  the merged assembly (`result.reactants` is `None`); `BondReactReacter`
  (which needs a pre-reaction snapshot for `fix bond/react` template
  generation, `_needs_reactants_snapshot = True`) takes exactly one.
- **Adjacency reuse** — `TopologyDetector.detect_and_update_topology`
  builds an atom → neighbors adjacency map once per call
  (`molpy.reacter.utils.build_adjacency`, O(bonds)) and threads it through
  every neighbor query, so angle/dihedral/improper enumeration is
  O(degree) per query instead of O(bonds).
  `find_neighbors` / `get_bond_between` / `count_bonds` accept the map via
  their `adjacency=` parameter; without it they fall back to the O(bonds)
  full scan.
- **Port registry** — the build loop tracks live port atoms per monomer
  node in a registry, remapped through each connection's entity map.
  Port lookup at connection time is O(live ports); the growing chain is
  never rescanned (`get_ports` / `get_ports_on_node` remain O(atoms) and
  are used only for the initial per-monomer scan).
- **Group map** — monomer-to-structure membership uses a group-id map
  with smaller-into-larger union instead of per-edge identity scans over
  all nodes.
- **Vectorized placement** — `Placer._apply_transform` applies
  `(coords - pivot) @ R.T + pivot + t` as one (N, 3) NumPy operation.

What still scales with chain length per connection: the reacter's input
copy of the accumulated structure (`_prepare_reactants`) and the merge
itself — O(chain) each, giving O(N²) total copying for a DP=N chain.
Eliminating that requires an in-place assembly mode and is out of scope
here (see `builder-reacter-05-perf` spec, Out of scope).

Counting-based performance tests live in
`tests/test_reacter/test_perf_copy_semantics.py` and
`tests/test_builder/test_polymer_build_perf.py` (a DP=200 wall-clock smoke
test is marked `@pytest.mark.slow`).
