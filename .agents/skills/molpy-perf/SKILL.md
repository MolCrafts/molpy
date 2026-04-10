---
name: molpy-perf
description: Performance profiling and NumPy optimization review. Use for performance-critical code like compute operators, builders, and I/O for large systems.
argument-hint: "[path or module]"
user-invocable: true
---

Review performance for: $ARGUMENTS

If no path given, check all files modified in `git diff --name-only HEAD`.

**Checks**

1. **Unvectorized loops**
   - Python for-loops over atoms, bonds, or frames → use NumPy vectorization
   - Per-atom distance calculations → `np.linalg.norm(r, axis=1)`
   - Per-pair operations → broadcasting or `scipy.spatial.distance`

2. **Memory anti-patterns**
   - Unnecessary `.copy()` of large coordinate arrays
   - Materializing full pairwise distance matrices for large systems
   - Loading entire trajectory into memory (should stream)
   - Large intermediate arrays not deleted after use

3. **Algorithm complexity**
   - O(N²) neighbor search → cell list or KD-tree for O(N)
   - Repeated sorting that could be cached
   - Redundant topology traversals (use igraph)

4. **I/O bottlenecks**
   - Text-based parsing for large files → consider binary formats (HDF5)
   - Repeated file open/close in loops
   - No buffering for large writes

5. **Lark parser optimization**
   - Grammar compiled once and cached (not per-call)
   - Transformer avoids creating unnecessary intermediate objects

6. **NumPy best practices**
   - `np.empty` + fill instead of `np.zeros` when values are immediately overwritten
   - Contiguous memory layout for iteration-heavy arrays
   - `np.einsum` for complex reductions

**Output format**:
```
PERFORMANCE REVIEW: <path>

✅ <check passed>
⚠️ [SEVERITY] Line N: <issue> — <recommendation>
❌ [SEVERITY] Line N: <issue> — <recommendation>

N ERRORS, M WARNINGS
```
