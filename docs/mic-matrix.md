# Unified periodic geometry and minimum-image distances

Validated locally on 2026-10-17 against ASE 3.29.0, freud-analysis 3.6.1,
and NumPy 2.5.4. Inputs use seed `20261017`.

## Contract and implementation

A cell has one matrix `H`, with lattice vectors as **columns**, an origin,
and a per-lattice-vector PBC mask. Cartesian/fractional conversion and
primitive-cell wrapping use that matrix for every geometry. Orthogonal boxes
are ordinary diagonal inputs. `BoxKind` remains descriptive metadata for
consumers with restricted shape domains; geometry kernels do not dispatch on it.

Primitive-cell wrapping and nearest-image selection have separate contracts:

- Wrapping returns `r - H m` inside the chosen primitive cell and returns `m`
  in the **original** lattice basis.
- MIC minimizes `|d - B n|` over integer `n`, where `B` contains only periodic
  lattice vectors. It imposes no cutoff prerequisite. With no periodic vectors,
  the same kernel has zero projection and one zero candidate, and returns `d`.

`Mic` is one prepared structure, rather than separate Free/Ortho/Triclinic
algorithm variants. `Mic::new` performs preparation once when the box is built
or replaced. The preparation is shared by `SimBox` and PME; callers cannot
construct a specialized kernel by filling enum fields.

Preparation sorts and reduces the periodic basis using integer column
operations. For rank at most three, exit checks the low-dimensional Minkowski
inequalities: for sorted vectors, no `b_j + sum(c_i b_i)` with `i < j` and
`c_i` in `{-1,0,1}` is shorter than `b_j`. Numerical comparisons use a relative
64-epsilon margin. A construction iteration limit returns an explicit error;
it does not fall back to an unchecked runtime algorithm.

The kernel caches the reduced basis, its dual, and at most `3^rank` image
translations. The dual is formed from a normalized Gram matrix to avoid
squaring the condition number due solely to unequal vector lengths. For partial
PBC it is the left inverse of the **periodic subspace**, not selected rows of
the full cell inverse.

Each `apply` projects the displacement, folds its periodic coefficients, evaluates
the cached translations, and selects the minimum squared periodic residual.
The perpendicular residual is constant across images and is excluded from
comparison, preventing a large nonperiodic component from hiding a small
periodic difference. There is no per-pair reduction, QR, recursion, heap
allocation, or data-dependent integer search. Equidistant images need not have
identical vector representatives across libraries.

ASE's general implementation similarly uses a reduced lattice and neighboring
image translations. Its reduction procedure differs; we compare the resulting
vectors rather than requiring the same reduced basis or reduction history.
See the official [general MIC implementation](https://docs.ase-lib.org/_modules/ase/geometry/geometry.html)
and [reduction conditions](https://docs.ase-lib.org/_modules/ase/geometry/minkowski_reduction.html).

## Numerical comparison

| Comparison | Coverage | Result |
|---|---|---|
| ASE `general_find_mic`, full PBC | 84 cells × 256 vectors = 21,504 | Maximum vector-component error `7.815970093361102e-14`; maximum distance error `6.394884621840902e-14` |
| ASE public `find_mic`, same random campaign | 21,504 vectors | No distance disagreements above `1e-9` |
| ASE `wrap_positions(eps=0)` | Same 21,504 vectors | Maximum component error `2.2737367544323206e-13`; original-cell image flags also round-trip |
| Independent exhaustive images, every PBC mask | 16 rotated cells × 8 masks × 64 vectors = 8,192 | Maximum component error `7.105427357601002e-15` |
| freud primitive-cell `wrap` | 32 cells × 256 vectors = 8,192 | Maximum component error `9.064080686016496e-6` |
| freud wrapping and distances inside the common safe cutoff | 32 cells × 128 vectors = 4,096 | Maximum vector-component error `7.817531837872593e-6`; distance comparison also passes |
| molrs LinkCell, ASE neighbor list, freud AABBQuery | 32 systems of 128 particles | Identical pair sets: 3,263 unique unordered neighbor pairs |

The full-PBC campaign includes diagonal cells, rotations, left-handed bases,
unreduced shears, and a shear of 1000. Unwrapped displacements are constructed
from fractional coordinates in `[-4,4]`. The exhaustive comparison constructs
finite integer bounds from the raw periodic basis's dual and an initial
candidate radius; it does not reuse molrs's reduction or candidate selection.

The freud common-domain campaign uses upper-triangular cells with tilt factors
in `[-0.25,0.25]` and cutoff `0.2 * min(nearest_plane_distance)`. Its small
residual discrepancies are consistent with freud's float32 implementation
versus molrs's float64. ASE accepts lattice vectors as rows, so it receives
`H.T`; freud's centered-cell wrapping is compared with molrs origin
`-H @ [0.5,0.5,0.5]`.

## Non-agreements outside common conditions

These are recorded explicitly rather than changing molrs's minimum-distance
contract to reproduce another implementation's shortcut.

### freud wrapping is not a global closest-image solver

For

```
H = [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]
d = [0.735, 0.49, 0]
```

molrs and ASE's general MIC return `[-0.265, 0.49, 0]` (distance about
`0.5570682184`). freud wraps to approximately `[0.735, 0.49, 0]` (distance
about `0.8833600625`). There are 1,130 such distance disagreements above
`1e-4` among the 8,192 arbitrary freud campaign vectors. The common safe-cutoff
and primitive-cell wrapping comparisons pass independently.

This follows freud's [source algorithm](https://github.com/glotzerlab/freud/blob/v3.6.1/freud/box/Box.h):
`computeDistance` uses fractional-cell `wrap`. It is not evidence of float
precision loss and is not treated as a molrs failure.

### ASE public shortcut on an unreduced basis

For `H = [[100,99,0],[0,1,0],[0,0,100]]` and `d = [1.48,-0.48,0]`,
molrs and ASE `general_find_mic` return `[0.48,0.52,0]`. Public `find_mic`
selects its naive shortcut and returns approximately `[1.48,-0.48,0]`.
This curated counterexample is separate from the random campaign's zero
public-API disagreements.

### ASE on oblique partial PBC

ASE's general routine differs from independent exhaustive results in 356 of
8,192 partial/full/open mask campaign vectors. The first counterexample,
including the original matrix, mask and all three results, is preserved in
[mic-comparison-results.json](mic-comparison-results.json). Molrs agrees with
the independently bounded enumeration for all vectors. The key distinction is
projection into the periodic subspace when nonperiodic cell vectors are oblique.

## Frozen evidence and default tests

[mic-comparison-results.json](mic-comparison-results.json) records versions,
counts, errors and counterexamples. `crates/molrs/src/core/simbox_oracle.json` contains
114 MIC and 80 wrapping vector goldens with their input cells and tolerances.
It includes both counterexamples above and an independently enumerated partial
PBC counterexample.

The comparison ran in a temporary, isolated Python environment. Per AGENTS.md,
scientific reference packages and the offline generator are not added to the
default gate or committed as runtime test dependencies. Rust unit tests read
only the frozen JSON, check the reduction inequalities, and check independent
hand-written image cases, large shear and partial PBC.

## Rust API migration

Direct construction of `Mic::{Free,Ortho,Triclinic}` is replaced by
`Mic::free()`, `Mic::ortho(lengths)` or `Mic::new(H,pbc)`. The first two
construct the same prepared representation. `simplified()` is removed because
there is no backend variant to simplify. The known sibling molpack checkout
still pins molrs 0.14 and uses this old method; its caller must drop that method
when migrating to the 0.18 API. No molpack dependency pins were changed here.

The accompanying core fixes also make `BlockView::to_owned()` and
`FrameView::to_owned()` return `Result<_, BlockError>`: callers must propagate
or handle malformed borrowed columns and metadata. Previously those errors
could silently drop data. `Atomistic::term_arity(block)` exposes the declared
arity needed to preserve empty custom terms in Python pickles. Structural
identity now reads `formal_charge`; partial `charge` is not an identity label.

## Verification before polish

- `cargo mrs-test --quiet`: 4,200 passed, 1 existing ignored test.
- `cargo mrs-doctest --quiet`: 151 passed, 10 ignored examples across the two rustdoc runs.
- `cargo mrs-clippy -- -D warnings`: passed.
- `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets -- -D warnings`: passed.
- `cargo clippy --manifest-path molcxx/Cargo.toml --all-targets -- -D warnings`: passed.
- `cargo fmt --all --check` and the Python binder's formatting check: passed.
- molrs Python suite: 1,782 passed, 6 skipped; the existing invalid-schema-cast test emits one NumPy RuntimeWarning, also present at baseline.
- molpy Python suite: 1,067 passed, 36 skipped.
- molpy core/tests Ruff checks and formatting: passed.

These are correctness and build checks, not a claim of benchmarked throughput.
The scientific comparisons used the locally rebuilt Python extension. No
dependency-version bumps or releases are part of this change.

## Polish review

Scope: molrs `4e51c90b`, molpy `170512d`, and their direct callers.

- Reuse / duplication: observable construction now directly calls the existing
  `molrs-python/src/core/block.rs::column_from_py`; its forwarding-only helper
  was removed. Canonical labeling and whole-graph isomorphism share the existing
  sorted parallel-edge representation, now owned by `GraphView::edge_bundles`.
  The topology-only subgraph matcher deliberately collapses parallel edges;
  its semantics differ, so it retains its separate projection while sharing
  the feasibility check.
- Placement: Python dtype conversion stays at the binding boundary, and graph
  label/multiplicity handling stays in the graph snapshot. No dependency or
  public API was added by polish.
- Leftovers / API: old MIC variants and `simplified()` have no remaining
  source callers inside molrs or its binders. Migration notes above also cover
  fallible borrowed-view conversion. The separately versioned molpack migration
  remains outside this change.
- Fixed: the exact graph matcher formerly collapsed parallel edges, allowing
  a four-cycle with edge multiplicities `[1,3,1,3]` to match `[2,2,2,2]`.
  A new regression failed before the fix. A second test checks that changing
  the insertion order of parallel-edge labels preserves isomorphism.
- Fixed: time monotonicity now compares neighboring values directly. The
  former subtraction overflowed for finite times `[-1e308,1e308]` and the
  reverse sequence, raising a warning instead of the documented `ValueError`.
  Both new cases failed before the fix. An additional regression covers finite
  time-bin indices above the int64 range.

Verification during polish:

- Baseline `cargo mrs-test --quiet -- core::`: 1,055 passed.
- Baseline binding `python -m pytest -q tests/test_trajectory.py
  tests/test_graph_hash.py tests/test_pickle.py tests/test_schema.py`: 52 passed.
- Baseline molpy `python -m pytest -q tests/test_core/test_selector.py
  tests/test_core/test_trajectory.py`: 49 passed; after the fixes, 52 passed.
- `cargo mrs-test --quiet -- core::` after the first graph regression/fix:
  1,056 passed; the final full run includes both graph regressions.
- Final `cargo mrs-test --quiet`: 4,202 passed, 1 existing ignored test.
- `maturin develop --locked --manifest-path Cargo.toml` in `molrs-python`:
  rebuilt and installed the extension successfully.
- Binding `python -m pytest -q`: 1,782 passed, 6 skipped, the same
  intentional invalid-schema-cast NumPy warning as baseline.
- Binding `python -m mypy python/molrs/_native.pyi` and
  `python -m mypy.stubtest molrs._native --mypy-config-file pyproject.toml
  --allowlist stubtest-allowlist.txt --ignore-disjoint-bases`: passed.
- molpy `python -m pytest -q`: 1,070 passed, 36 skipped.
- molpy `ruff check src/molpy/core tests/test_core` and
  `ruff format --check src/molpy/core tests/test_core`: passed.
- Root `cargo fmt --all --check`, binding
  `cargo fmt --manifest-path molrs-python/Cargo.toml --check`, and both
  repositories' `git diff --check`: passed. A formatting invocation initially
  used this root-relative manifest from the binding directory and failed to
  locate it; the corrected invocation from the root passed.
- `cargo clippy --manifest-path molrs-python/Cargo.toml --all-targets
  -- -D warnings`: passed.
- `cargo mrs-clippy -- -D warnings`: passed.

Locations (workspace-relative):

- `molrs/molrs/src/core/graph_hash.rs:129` owns the shared edge bundles;
  canonical labeling calls it at line 333, isomorphism at lines 516–517;
  regressions start at lines 749 and 770.
- `molrs/molrs-python/src/core/trajectory.rs:238` calls
  `molrs/molrs-python/src/core/block.rs:1353` directly.
- `molpy/src/molpy/core/_splitter.py:95` avoids subtraction;
  regressions are `molpy/tests/test_core/test_trajectory.py:119` and line 124.
- `molrs/docs/mic-matrix.md:143` documents borrowed-view API migration.

Ready: yes, for the two reviewed repositories. No material issue remains in
this scope. The earlier CXX and doctest results above remain applicable;
polish changes no exported signature or MIC implementation. The offline
ASE/freud campaigns were not rerun during polish.

Next: none.
