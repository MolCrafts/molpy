# Performance Standards

MolPy-specific NumPy / algorithm performance rules. Consumed by `/mol:review --axis=perf`
and the `optimizer` agent. Migrated from the former local `molpy-perf` skill and
`molpy-optimizer` agent (2026-06-10).

## Hot paths (profile these first)

- **Compute operators** (RDF, MSD, order parameters, …): called per-frame, must be fast.
- **Pairwise distance**: the most common inner loop across compute/ and builder/.
- **Builder placement**: native (`TracePlacer`); on the molpy side, keep it to
  one `place` call per assembly batch.
- **Parser / SMARTS**: SMILES/SMARTS run in molrs — avoid re-compiling the same
  SMARTS pattern in a hot loop; reuse `SmartsPattern` instances.

## Vectorization rules

- No Python for-loops over atoms, bonds, or frames — use NumPy vectorization.
- Per-atom distances: `np.linalg.norm(r, axis=1)`, not a loop.
- Per-pair operations: broadcasting or `scipy.spatial.distance`.
- `np.einsum` for complex tensor contractions/reductions.
- `np.add.at` for scatter operations.
- `np.empty` + fill instead of `np.zeros` when values are immediately overwritten.

## Memory rules

- No unnecessary `.copy()` of large coordinate arrays; prefer views (boolean masks)
  over fancy indexing where possible.
- Never materialize full pairwise distance matrices for large systems.
- Stream large trajectories — never load-all-into-memory.
- `del` large intermediates in multi-step calculations.
- `np.float32` for coordinates is acceptable when float64 precision is not needed.
- Contiguous memory layout for iteration-heavy arrays.

## Algorithm complexity

- Neighbor search: O(N) cell lists or KD-tree, never O(N²) all-pairs for large N.
- Topology/graph algorithms: use the molrs graph kernels (`Atomistic` topology,
  `NeighborQuery`), do not hand-roll traversals; cache repeated traversals.
- Avoid repeated sorts; maintain sorted invariants or cache sort results.
- Document complexity in docstrings: O(N), O(N²), etc.

## I/O

- Text parsing for large files is a bottleneck — prefer binary formats (HDF5).
- No repeated file open/close in loops; buffer large writes.

## Discipline

- Never sacrifice correctness for speed. There is no benchmark harness in this
  repo (the benchmark/regression system is being redesigned); a performance
  change is verified for correctness by the unit suite and its speed claim is
  recorded here as owed, not asserted.

## Profiling commands

```bash
python -m cProfile -o profile.out script.py
kernprof -l -v script.py                      # pip install line_profiler
python -m memory_profiler script.py           # pip install memory_profiler
```

## Owed (2026-09-20)

Every hot-path finding from the 0.14 cleanup has been acted on; the list is
kept so the reasoning is not lost.

- `builder/assembly/_proximity.py` — no-cutoff site pairing computes all
  distances in one numpy broadcast (the O(sites_a × sites_b) *output* is the
  pairing itself); connected components come from one `topo_distances`
  traversal per component; chain-end degree from `incident_relations`.
- Placement — the molpy `builder/assembly/_placer.py` is gone. Placement is
  the native `TracePlacer` (with `Trace` / `LineOrienter` / `TangOrienter`),
  re-exported by identity; `GraphAssembler.apply` makes one `place(world,
  bonds)` call per batch, so there are no per-atom PyO3 calls on the molpy
  side. Its performance is owned and measured in molrs.
- `builder/assembly/_assembler.py` — `_total_charge` sums the charge column
  when every atom carries one; a partial column is summed entity-wise over its
  validity mask (unblocked by molrs `column()` raising on holes instead of
  zero-filling).
- `core/atomistic.py` — `symbols` reads the element column. `select` takes a
  per-atom Python predicate by contract, so it is O(N) Python by design;
  `get_neighbors` is already O(degree) through the adjacency index.
  `def_*s` still call Rust once per element: a batch entry is a molrs feature
  request, not a molpy fix.
- `adapter/rdkit.py` — single pass each way, joined by the `mp_id` tag; the
  `id`/`mp_id` reconciliation passes are gone (rdkit is optional; tests under
  `tests/test_adapter/test_rdkit.py` skip without it).
- `typifier/clp.py` — `clp.xml` is parsed once, by the native typifier.
- `pack/constraint.py` — deleted with `molpy.pack` in 0.14; packing (and its
  penalty kernels) is the external molpack package.
- `compute/dielectric.py` — deleted in 0.15 together with the
  `DielectricSusceptibility` / `IonicConductivity` recipe classes; the
  dielectric primitives are native re-exports on `molpy.compute`, so there is
  no molpy dielectric hot path left.
