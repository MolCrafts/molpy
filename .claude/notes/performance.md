# Performance Standards

MolPy-specific NumPy / algorithm performance rules. Consumed by `/mol:review --axis=perf`
and the `optimizer` agent. Migrated from the former local `molpy-perf` skill and
`molpy-optimizer` agent (2026-06-10).

## Hot paths (profile these first)

- **Compute operators** (RDF, MSD, dielectric, …): called per-frame, must be fast.
- **Pairwise distance**: the most common inner loop across compute/ and builder/.
- **Builder placement**: coordinate transforms in tight loops.
- **Parser**: Lark grammar compilation — compile once and cache, never per-call;
  transformers should avoid unnecessary intermediate objects.

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
- Topology/graph algorithms: use igraph, do not hand-roll traversals; avoid
  redundant topology traversals (cache them).
- Avoid repeated sorts; maintain sorted invariants or cache sort results.
- Document complexity in docstrings: O(N), O(N²), etc.

## I/O

- Text parsing for large files is a bottleneck — prefer binary formats (HDF5).
- No repeated file open/close in loops; buffer large writes.

## Discipline

- Never sacrifice correctness for speed; benchmark before and after every change.

## Profiling commands

```bash
python -m cProfile -o profile.out script.py
kernprof -l -v script.py                      # pip install line_profiler
python -m memory_profiler script.py           # pip install memory_profiler
```
