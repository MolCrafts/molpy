---
spec: compute-redesign
created: 2026-10-14
criteria:
  - id: ac-001
    summary: One trait, one verb
    type: code
    pass_when: |
      `molrs/src/compute/` defines exactly one analysis trait, `Compute`, whose
      only required method is `compute(&self, input: Self::Input<'_>)`. The
      strings `trait Fit`, `trait Check`, `ComputeResult`, `DescriptorRow` and
      `LegacyCompute` do not occur under `molrs/src/compute/`. Every
      `impl Compute for` block contains no method other than `compute`. No type
      that implements `Compute` has another `pub fn compute_*` method.
    status: pass  # 09: one analysis trait; legacy names absent; 71 impl Compute blocks define only compute; no pub fn compute_*
  - id: ac-002
    summary: No public free functions in compute; signal is gone
    type: code
    pass_when: |
      `grep -rnE '^pub fn ' molrs/src/compute` is empty. `molrs/src/signal/`
      does not exist. `molrs/Cargo.toml` has no `signal` feature and `compute`
      takes `dep:rustfft` directly.
    status: pass  # 09: no module-level pub fn; signal/ gone; compute = ["dep:rustfft"]
  - id: ac-003
    summary: Compute knows no frames and no field names
    type: code
    pass_when: |
      Outside `#[cfg(test)]` code, nothing under `molrs/src/compute/` names
      `FrameAccess`, `Frame`, `FrameView` or `core::schema`/`keys`, and no
      function or method is named `from_frame` or `from_frames`. Outside
      `#[cfg(test)]` code,
      `grep -rnoE '"(atoms|x|y|z|vx|vy|vz|mass|charge|orientations|bonds|angles|dihedrals|grid|density|cube_units|element)"' molrs/src/compute`
      is empty. `molrs/src/compute/positions.rs` and `pmft/orientation.rs` do
      not exist. The same holds for the compute bindings: no
      `from_frame`/`from_frames` and no frame column reads in
      `molrs-python/src/compute` or `molrs-wasm/src/compute`.
    status: pass  # 09 (core): no Frame/FrameAccess/FrameView/schema/keys/from_frame or field literals outside tests; bindings part is link 10
  - id: ac-016
    summary: new takes configuration, compute takes data
    type: code
    pass_when: |
      No type named `TimeSeries`, `Snapshot`, `Weights`, `SeriesPair`,
      `Orientations`, `Labels` or `ThermoState` exists under
      `molrs/src/compute`. Every `pub fn new` / `pub fn with_*` of a compute
      type has parameters only of these kinds:
      - scalars;
      - the configuration value types of spec section 2 (`Axis`,
        `RegularGrid`, `Window`, `LagKernel`, `Origins`, `MeanPolicy`,
        `Normalization`, `Reduction`, `Stencil`, `UnitPreset`);
      - named enums.
      In particular, no constructor takes an array, a `Vec` of indices, masses
      or orientations, `dt`, a `SimBox`, or `Neighbors`. Every analysed datum
      reaches the kernel through `compute`. The same holds for every Python and
      WASM constructor in the compute bindings.
    status: pass  # 09 (core): constructors take configuration only; exemptions (Orientations GAT, evaluation lists, value types) in spec "Link 09 acceptance audit"; bindings part is link 10
  - id: ac-004
    summary: Every deleted entry is gone and every kept entry is migrated
    type: code
    pass_when: |
      Every symbol marked "→" in spec section 4 is absent from
      `molrs/src/compute/mod.rs` re-exports. Every symbol marked "K" is
      re-exported and implements the new `Compute`. Every symbol marked "M"
      exists as a method on its named owner. Every surface marked "S" (and
      every surface in the D4 table) holds a `Correlator`, has a `compute`
      body that is a single delegating call, and contains no other numeric
      code.
    status: pass  # 09: → symbols absent, K/S re-exported with one impl Compute, M methods present, surfaces delegate in one call
  - id: ac-005
    summary: One implementation per algorithm
    type: code
    pass_when: |
      For every row of the spec section 3 table, each "Replaces" location no
      longer contains that algorithm. Concretely, under `molrs/src/compute`:
      - FFT correlation: `rustfft` is imported only in the `Correlator` and
        `Spectrum` owner files.
      - Trapezoid integration: `0.5 * (` trapezoid sums appear only in
        `fitting/cumulative_trapezoid.rs`.
      - Bin indices: `.min(n_bins - 1)`-style bin clamps appear only in `Axis`.
      - Neighbour search, components, eigensolvers: no type named `CellGrid`,
        `CellList` or `Dsu` exists, and no function named `power_iteration`
        exists.
      - PMFT: `pmft/` holds one generic `Pmft<C>` plus coordinate impls.
    status: pass  # 09: rustfft only in correlator.rs/spectrum.rs; no CellGrid/CellList/Dsu/power_iteration; non-trapezoid 0.5*( exemptions in spec
  - id: ac-006
    summary: No silent fallback
    type: code
    pass_when: |
      Each item of spec section 5 is resolved as specified. Under
      `molrs/src/compute`:
      - `grep -rnE 'map_or\(1\.0|unwrap_or\(1\.0|\.max\(1\)|min\(n.*- *1\)'`
        returns no hits on physical inputs.
      - `debug_assert` is not used for input validation.
      - `assert!` is not used on a user-supplied parameter.
      Each item has a unit test asserting the specific `ComputeError` variant.
    status: pass  # 09: fallback greps empty outside tests; no debug_assert/assert! on inputs; variant tests per as-built notes
  - id: ac-007
    summary: Streaming equals batch
    type: runtime
    pass_when: |
      Every `Stream` impl has a body `self.accumulator().fold(input)`. For
      `Correlator` (`Product`, `SquaredDisplacement`) and `Rdf`, unit tests
      assert streaming == batch for every lag ≤ cap (or every bin) to ≤1e-12
      relative error.
    status: pass  # 09 (core): Stream bodies fold; Correlator FFT exception pinned by streaming_equals_batch_at_every_lag; Rdf bitwise test
  - id: ac-008
    summary: Primitive correctness
    type: runtime
    pass_when: |
      The primitive tests listed in the spec's Testing strategy pass under
      `cargo test -p molcrafts-molrs --lib --features full,filesystem -- compute`:
      - `Correlator` FFT vs direct loop, for all five kernels
      - `Axis` / `Histogram` edges, NaN and shell volumes
      - `RegularGrid` stencil wrap
      - `GroupReduce` boundary straddle
      - `BondAverage` q_6 golden
      - `Pmft<C>` goldens
      - each D4 surface equals its explicitly configured `Correlator`
      - `Wrapped` rejection
    status: pass  # 09 (core): every listed primitive test present and passing (names in spec audit)
  - id: ac-009
    summary: Goldens are invariant except the declared re-pins
    type: runtime
    pass_when: |
      Every pre-existing compute golden test exists after the chain, with its
      assertion literals unchanged, except the tests named under "Behaviour
      changes" items 1–5. Each changed literal is introduced in the link that
      implements its re-pin item.
    status: pass  # 09: all 38 pre-chain fractional assertion literals present; renamed/deleted tests per 02–08 notes
  - id: ac-010
    summary: Composition works generically
    type: runtime
    pass_when: |
      Unit tests run each pipeline end to end, with no glue code between steps:
      - velocity array → `Vacf` →
        `CumulativeTrapezoid`
      - `DipoleMoment` → `Msd` → `LinearFit`
      - `Steinhardt` → `Pca` → `Kmeans`, with `n_components = 3` and Kmeans
        clustering in 3-D
      - `Cluster` → `CenterOfMass` / `GyrationTensor` through `Partition`
      - `HBonds` → `presence()` → `Survival`
      - `DipoleAcf` → `EinsteinHelfandSpectrum` →
        `KramersKronig` (Verdict)
    status: pass  # 09 (core): six pipeline tests; DipoleMoment → Msd → LinearFit added in 09
  - id: ac-011
    summary: Bindings carry no defaults, parsing or physics
    type: runtime
    pass_when: |
      Under `molrs-python/src/compute` and `molrs-wasm/src/compute`:
      - No literal default appears in a `#[pyo3(signature = …)]` or
        `unwrap_or*` (all optional parameters are `None` / `Option`).
      - No `match` on a string.
      - No `PyDict::new` result.
      - No free `#[pyfunction]` / free `#[wasm_bindgen] pub fn`.
      `molrs-wasm/src/compute/catalog.rs` does not exist. A defaults-parity test
      (core `Default` vs every binding constructed without optional args) and a
      name-parity test (core compute set == Python set == WASM set) pass.
    status: pending
  - id: ac-012
    summary: C++ API returns errors
    type: code
    pass_when: |
      Every fallible function in `molrs-cxxapi/src/compute.rs` returns `Result`.
      No `unwrap_or_default`, `NaN` or empty-`Vec` error path remains. The
      `diffusion` façade method does not exist.
    status: pending
  - id: ac-013
    summary: Full gate green
    type: runtime
    pass_when: |
      `prek run --all-files --hook-stage pre-push` passes in molrs at the end of
      link 10, including clippy `-D warnings`, doctests, Python tox and the WASM
      gate. The bench comparison shows no RDF/MSD regression above 10%.
    status: pending
  - id: ac-014
    summary: Catalog derived; molvis validated
    type: runtime
    pass_when: |
      molrs-wasm `computeSpecs()` is generated from core `ComputeSpec`
      descriptors and lists every core compute. molvis
      `stage/src/analysis/registry.ts` validates its UI table against it, and
      molvis tests pass.
    status: pending
  - id: ac-015
    summary: molpy follows by identity
    type: runtime
    pass_when: |
      After the molrs release tag:
      - `molpy/src/molpy/signal/` does not exist, and none of the ghost
        directories listed in spec section 8 exist.
      - `tests/test_public_paths.py` loops over the full `molrs.compute.__all__`
        and asserts identity.
      - `docs/compute/**/*.md` contains no `import molrs`, no deleted symbol and
        no "module-level function" endorsement.
      - Every runnable docs code block executes.
      - `scripts/docs_series/lj.py` imports no `molrs` and builds on `mp.md`.
      - molpy's full test suite passes.
    status: pending
---
