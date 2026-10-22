# Consuming molrs from another project

molrs (`molcrafts-molrs`) exposes its data and force-field types through **two
as-built paths**. Pick by what your project is:

| Your project | Path | Crate | Cost |
|---|---|---|---|
| A Rust crate / binary | **Native** | `molcrafts-molrs` (direct dep) | zero-copy by construction |
| A Python / WASM binding | **Handle API** | `molrs-ffi` | zero-copy column borrows across the language boundary |

There is no marshalling layer and no `to_dict`/`from_dict` round-trip — a consumer
holds molrs data directly (native) or through a stable handle (FFI). The reference
Rust consumer, [`molcrafts-molpack`](https://github.com/MolCrafts/molpack), uses the
native path: its `Cargo.toml` depends on `molcrafts-molrs` directly and operates on
`molrs::core::Frame` / `molrs::ff::forcefield::ForceField` natively.

---

## Path A — native Rust (depend on `molcrafts-molrs`)

Add the crate, enabling only the sub-systems you need (`core` is always on).
Downstream packages that co-release with molrs (e.g. molpy) pin the shared
**major.minor** line (`>=X.Y.0,<X.(Y+1)`), not an exact patch.

```toml
[dependencies]
molrs = { package = "molcrafts-molrs", version = "0.19", features = ["ff"] }
```

Then use the native types directly — no FFI, no copies. For example, building
evaluable MMFF94 potentials from a molecule (the pattern molpack's relaxer follows):

```rust,no_run
use molrs::core::Atomistic;
use molrs::ff::compile::PotentialCompiler;
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::Typing;
use molrs::ff::typifier::mmff::Mmff94Typifier;
// UFF: use molrs::ff::typifier::UffTypifier  (same composition)

let mol = Atomistic::new();                              // build or load your molecule
let mut typing = Typing::new(Mmff94Typifier::new());

let mut frame = typing
    .typify(&mol)?
    .to_frame()
    .map_err(|e| e.to_string())?;                        // labels + charges
let ff = typing.forcefield();                            // exactly the types assigned
// The consumer's neighbour list — built from the force field's own
// special_bonds, which decide whether 1-2 / 1-3 neighbours belong in it.
frame.insert("pairs", intramolecular_pairs(&frame, ff.special_bonds())?);
let potentials = PotentialCompiler::new(ff).compile(&frame)?; // the standard compile path

let coords: Vec<f64> = Vec::new();                       // flat [x,y,z, ...]
let (energy, _forces) = potentials.calc_energy_forces(&coords);
println!("MMFF94 energy = {energy} kcal/mol");
# Ok::<(), String>(())
```

There is no MMFF/UFF shortcut, and that is the point: a force field read from a
file is consumed by exactly these three lines. A typifier implements only
`assign`; `Typing` wraps it, and `Typing::typify` — labels and charges — is the
only writer of the output `Typing::forcefield()`, which holds exactly the
definitions typing assigned. Compiling is
`PotentialCompiler::new(ff).compile(&frame)`.
Python and WASM bind the same composition, with no shortcut either: a
typifier class is named after the Rust typifier (`Mmff94Typifier`,
`UffTypifier`, …) and folds in its `Typing` driver — `typify(…)` and
`forcefield()` — and `PotentialCompiler(forcefield).compile(frame)` compiles
(JS: `new PotentialCompiler(typifier.forcefield()).compile(typed)`).
The neighbour list is *yours* because you are the one who knows when it goes
stale: a minimizer that moves atoms decides when to rebuild it, and molrs will
not guess. (WASM `Lbfgs` takes its pairs from the `Neighbors` table it is
constructed with, through `ff::potential::intramolecular_pairs_from_neighbors`
— the same exclusion and 1-4 rules as `intramolecular_pairs`.)

(The three steps stay separate on purpose: the typed `Frame` between them is
where a missing term, such as an absent electrostatic style, is visible. This
is the pattern the molpack relaxer follows.)

This exact snippet is compile-checked as the module doctest on
`molrs::ff::typifier::mmff`.

MMFF ships **two** named front doors — `Mmff94Typifier` and `Mmff94sTypifier` —
over one engine. Swap the type to swap the parameter set; there is no variant flag.
MMFF94s (Halgren 1999, the "static" set) re-parameterises 11 out-of-plane rows and
42 torsion rows so that delocalised trivalent nitrogen (MMFF types 10 `NC=O` /
40 `NC=C`) minimizes planar; everything else is shared, so a molecule without such
a nitrogen gets bit-for-bit identical potentials from both.

`UffTypifier` is the third named front door (RDKit-aligned Universal Force
Field). Same composition; no electrostatics.

## Path B — Python / WASM via the `molrs-ffi` handle API

Language binders (`molpy`, `molwasm`, `molc`, `molcxx`) link `molrs-ffi` and hold
a **handle** — a `FrameRef` (a `FrameId` paired with the shared `FrameArenaCell` that owns the frame) — forwarding
every column access through the shared helpers. Numeric columns are borrowed as
contiguous slices (zero-copy); strings are copied (they aren't contiguous scalars).

```rust,no_run
use molrs::op::Idx;
use molrs_ffi::FrameRef;

let frame = FrameRef::default();          // a frame inside a fresh FrameArenaCell
// ... populate it via frame.with_mut(|f| ...) ...
if let Ok(atoms) = frame.block("atoms") {
    // zero-copy borrow of the uint atom-id column (see the uint-index contract below)
    let n_ids = atoms.borrow::<Idx, _>("id", |ids, _shape| ids.len()).ok().flatten();
    let _ = n_ids;
}
```

`molrs-ffi` exposes `FrameRef`, `BlockRef`, `ForceFieldRef` (under the `ff` feature),
`RegionRef` (a shared `Arc<dyn Region>`), `FrameArena` and its shared cell `FrameArenaCell`, `FrameId`,
`BlockHandle`, and one error type `FfiError`.
This snippet is compile-checked as the `molrs-ffi` crate-level doctest.

### ABI contract (cross-extension handle exchange)

Two separately compiled extensions (e.g. the `molcrafts-molrs` wheel and the
`molcrafts-molpack` wheel) may exchange raw `molrs_ffi` handles through
PyCapsules. That is a pointer bridge, so both sides must embed a
**layout-identical** molrs core. The rule — decided project-wide — is:

> **Minor-line = ABI version.** Every downstream shares one molrs minor line.
> Within a minor line the layout of every FFI-crossing type is frozen; a
> layout change requires a minor bump. When molrs moves to a new minor,
> downstream is obliged to re-align.

`molrs_ffi::abi` is the single source of the contract; **never hard-code the
capsule names**:

- `abi::abi_line()` — `major.minor` of the embedded molrs (e.g. `"0.19"`).
- `abi::frameref_capsule_name()` / `abi::forcefield_capsule_name()` /
  `abi::regionref_capsule_name()` — `molrs.FrameRef/<line>` /
  `molrs.ForceFieldRef/<line>` / `molrs.RegionRef/<line>`. The line in the
  name means a cross-minor exchange fails the capsule *name check* — a clean `ValueError` — instead of
  dereferencing a possibly drifted layout.
- `molpy._ffi_abi_token()` (Python) — returns
  `(abi_line, version, frameref_name, forcefield_name, regionref_name)`. A
  consumer extension calls it once at import and raises a clear `ImportError`
  on a line mismatch (molpack's `interop::check_abi` is the reference
  implementation; it reads the first two entries, so the tuple may grow).

**Regions cross as geometry the consumer evaluates.** Every molpy region
object (`Sphere`, `Cuboid`, `Parallelepiped`, `HalfSpace`, `Cylinder`,
`Ellipsoid`, `Polyhedron`, `SphereUnion`, and a composed `Region`) exports
`_ffi_regionref_capsule()`: a capsule named `molrs.RegionRef/<line>` whose
`void*` is `*mut *mut RegionRef`. The consumer resolves it exactly like a frame
capsule — `capsule.pointer_checked(Some(abi::regionref_capsule_name()))`,
dereference twice, `.clone()` the handle — and keeps `handle.region()`, an
`Arc<dyn Region + Send + Sync>` it may share into a rayon loop. Unlike a frame,
the handle's *code* runs in the producer's image (vtable dispatch), so the
cross-image contract is the `[F; 3]` surface only: `distance`, `distance_grad`,
`contains_point`, `bounds` — none panics on finite input. The batched
`contains(&Fnx3)` can panic on a malformed array and is not part of it.

Enforcement on the supply side: `crates/molrs-ffi/src/abi.rs` carries a **layout
snapshot test** (size / align / field offsets of every FFI-crossing type,
committed as `src/layout.snapshot`). Changing any of those layouts within a
minor fails CI; a toolchain update that alone changes the report is treated
the same way (the bridge crosses compiled layouts, not source).

Version combinations:

| producer (molrs wheel) | consumer (e.g. molpack) | outcome |
|---|---|---|
| same minor, any patch | same minor, any patch | **supported** — layout frozen by the snapshot gate |
| line X | line Y ≠ X | `ImportError` at consumer import (token mismatch); a capsule resolved anyway fails the name check with a clean `ValueError` |

Release ordering is unchanged: molrs ships a new minor first; molpy / molpack
re-align and ship after ("Release before molpy" iron law).

---

## Path C — C ABI (`libmolc`)

The **only sanctioned dynamic-linking deliverable**. External C / C++ / HPC
consumers link `libmolc` (cdylib or staticlib) against the
cbindgen-generated `molc.h` — a flat, handle-based C API over frames,
blocks, boxes, force fields, and regions (feature surface: always-on core
+ perceive, plus `ff`, `io`, `smiles`; every object lives in one global,
mutex-protected handle registry, so treat the library as single-threaded per
process).

- **Download**: `molc-<version>-<platform>.tar.gz` (lib + `molc.h` +
  LICENSE + sha256) attached to each GitHub Release on `v*` tags.
- **Identity**: `molrs_version()` reports the embedded molrs release for
  diagnostics. There is no C or CXX API version handshake before 1.0;
  a breaking signature change ships as a new library, not a version gate.

### Regions across the boundary

A region is `Arc<dyn Region>` — a trait object — and it does **not** cross any
boundary. What crosses is `molrs_ffi::RegionRef`, the same handle the Python
capsule (`molrs.RegionRef/<abi_line>`) and the WASM binder carry; the C API
keeps it in its handle registry and hands back the usual two-word
`MolrsRegionHandle`. So a region is no different from a `SimBox` or a
`ForceField` at this seam, and the vtable stays on the Rust side where it was
compiled.

```c
MolrsRegionHandle outer, inner, hole, shell;
molrs_region_sphere((const double[3]){0, 0, 0}, 3.0, &outer);
molrs_region_sphere((const double[3]){0, 0, 0}, 2.0, &inner);
molrs_region_not(inner, &hole);
molrs_region_and(outer, hole, &shell);          /* a shell */

bool inside[1];
molrs_region_contains(shell, (const double[3]){2.5, 0, 0}, 1, inside);

molrs_region_drop(shell);                       /* operands stay alive */
molrs_region_drop(hole);
molrs_region_drop(inner);
molrs_region_drop(outer);
```

Three questions, one answer shape on every surface: `molrs_region_distance`
gives the signed distance (negative inside), `molrs_region_contains` is its
sign, and `molrs_region_bounds` writes `[xmin, xmax, ymin, ymax, zmin, zmax]`.
Composition — `and` / `or` / `not` — returns an ordinary handle, so
compositions nest, and each handle owns its own reference: dropping a
composition never disturbs its operands. A stale handle is reported as
`MolrsStatus::InvalidRegionHandle`, never dereferenced.

The CXX bridge carries the same surface as free functions over a
`Box<RegionRef>` (`region_sphere`, `region_and`, `region_distance`, …), gated
by the `CXX_CAP_REGION` capability bit so a consumer can fail loudly when it
is linked against a bridge without it. A vector argument that is not
exactly three values, or a ragged point list, throws `rust::Error`; nothing
falls back to the origin or to an empty answer.

The rest of the CXX bridge (`molcxx`, consumed by Atomiverse) names
each item after the molrs owner it fronts:

| Bridge item | molrs owner |
|---|---|
| `FrameRef`, `frame_new`, `frame_column_{f64,i32,u64,str}`, `frame_set_column_{f64,i32,u64,str}` | `core::Frame` via `molrs_ffi::FrameRef` (`u64` is the `UInt` / `Idx` dtype) |
| `frame_box_h` / `frame_set_box_h` | the box's cell matrix H, 9 row-major values (`SimBox::h_view`) |
| `frame_meta_keys`, `frame_get_meta`, `frame_set_meta`, `KeyedMetaValue` | `Frame::meta` (a key with its exact-dtype `MetaValue`) |
| `read_xyz`, `write_xyz` (one frame; `append` adds it after the frames already there) | `io::read_xyz`, `io::xyz::XyzWriter` |
| `read_mrec_frame`, `write_mrec_frame` | `io::read_mrec_frame`, `io::write_mrec_frame` (a record's `frame` section) |
| `read_mrec_trajectory_frame`, `MrecWriterRef` (`mrec_writer_create` / `open` / `append` / `flush` / `committed` / `close`) | `io::mrec::MrecReader::frame`, `io::mrec::MrecWriter` |
| `Msd` (`msd_new(max_lag)`, `set_origins`), `Vacf` (`vacf_new(max_lag)`, `set_normalization`, `set_stride`, `set_mean`) — `compute(...)` over a whole trajectory, `accumulator()` → `CorrelatorAccumulator` (`push` one frame, `n_frames`, `finish`) | `compute::Msd`, `compute::Vacf`, `compute::CorrelatorAccumulator` (`Stream` / `Accumulate`) |
| `TimeCorrelation` (`lag_times`, `n_lags`, `n_channels`, `channel`) | `compute::TimeCorrelation` |
| `LinearFit` (`linear_fit_new(start, end)`, `compute(correlation, channel)` → `LinearFitResult`) | `compute::LinearFit` (Einstein `D = slope / 2d` of an `Msd` curve) |
| `Rdf` (`rdf_new(n_bins, r_max, r_min)`, `compute(positions, boxes, n_frames)` → `RdfResult`), `accumulator()` → `RdfAccumulator` (`push(positions, h)`, `n_frames`, `finish`) | `compute::Rdf`, `compute::RdfAccumulator` over a `core::NeighborList` self-query |
| `BccModel` (`bcc_model_new`, `assign`, `correct`) | `ff::charge::BccModel` (`BccParameterSet::from_name`, `assign`, `correct`) |

The compute items follow the molrs contract: a constructor takes
configuration only, `compute` takes a whole trajectory and the accumulator
takes one frame per `push`. A trajectory crosses as a flat row-major
`[n_frames, D·n]` buffer, each frame blocked by component (`x|y|z` positions,
or `n_dof` velocity values), and a box as its 9-value H; an optional box is an
empty slice. Every fallible call returns `Result`, so a bad argument, a ragged
buffer or a compute error throws `rust::Error` — no call answers an error with
an empty curve, a NaN or `false`. There is no fused MSD-plus-fit call: the
diffusion coefficient is `Msd` → `TimeCorrelation` → `LinearFit`.

In-house Rust consumers (molpack, the binders) do **not** go through this C
ABI — they take Path A or Path B directly, and every one of them links molrs
statically.

---

## Data contract (both paths)

Whichever path you take, molrs data follows these conventions:

- **Atom indices are unsigned** (`u64`, the `UInt` dtype). Index columns —
  `atoms.id`, the `atomi`/`atomj`/`atomk`/`atoml` columns on bond/angle/dihedral
  blocks — are read via `block.get(key).and_then(|c| c.get::<Idx>())` (native) /
  `borrow::<Idx, _>` (handle). Do **not** read them as signed. Record readers refuse
  one stored at a narrower width.
- **Pairs block schema.** A non-bonded pair list is a block with `atomi`, `atomj`
  (uint) and `is_14` (bool) columns. This is the single pairs convention across the
  force field.
- **`special_bonds` weights live on the `ForceField`**, not in the neighbour list.
  The force field carries the 1-2 / 1-3 / 1-4 LJ and Coulomb scale factors
  (e.g. amber `0/0/0.5` LJ, `0/0/0.8333` Coulomb); a reader fills them.
- **`BondDistanceWeights` is the core geometric table** (Cassandra 1-N tail,
  always-on, one vector). It is not `ForceField::special_bonds`. A length-3
  vector here is a zero-or-full tail, not a LAMMPS triple: charmm `0 0 0`
  is `[0, 0, 0, 1]`. There is no `From`/`Into` between the two types.
- **The neighbour list is the consumer's job.** `ForceField` holds parameters +
  `special_bonds` only; the optimizer / integrator builds the intramolecular pair
  list (`molrs::ff::potential::intramolecular_pairs(&frame, ff.special_bonds())
  → atomi/atomj/is_14`) and inserts it before calling `PotentialCompiler::compile`. The
  weights are an argument because they decide the rows: `special_bonds fene`
  (`[0, 1, 1]`) keeps 1-3 pairs. A list of rows expresses a 1-2 / 1-3 weight of
  `0` or `1` and nothing else, so a force field that *scales* those classes —
  or scales them differently for van der Waals and Coulomb — is an `Err` here
  and belongs on `PotentialCompiler::compile_typed`, which carries a per-pair weight.

## Which path?

- Writing Rust → **Path A**. You get molrs types natively with no boundary cost;
  there is no reason to route through `molrs-ffi`.
- Writing a Python/WASM binding → **Path B**. Hold a `FrameRef`, borrow columns
  through `BlockRef`, and map names with your binding's attribute macros.
