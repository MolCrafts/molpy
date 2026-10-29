---
title: molpy.md calculator seam — CPU integrator, GPU force model, one cheap exchange per step
slug: md-calculator-seam
status: approved — 01–05 implemented 2026-10-29 (uncommitted)
created: 2026-10-29
chain: md-calculator-seam (01-step-hygiene → 02-units → 03-cadence → 04-calculator-core → 05-calculator-binding → 06-barostat → 07-dump-checkpoint → 08-energy-drift → 09-gate) → molnex (adapter, delete molix.md)
repos: molpy (01–08), out-of-source script (09), molnex (downstream)
---

# molpy.md calculator seam

## Summary

molnex's torch MD engine (`molnex/src/molix/md/`, 3.6k lines) is replaced by
molpy.md. Integration stays on the CPU in `crates/molrs/src/md/`. The force side
stays in molnex: the model, its compile / CUDA-graph build, its precision
policy, and its GPU fixed-capacity neighbour search
(molnex spec `neighbor-01-batched-capacity-search`). The two meet at an
ASE-shaped **calculator seam**. A calculator is bound once with the frame.
Each step it is handed positions, the cell and two flags (cell changed,
neighbours rebuilt). It returns energy, forces and, when asked for one, a virial.

The exchange between host and device has to be cheap. Neither side allocates
anything per step. The host buffers are pinned and supplied by the
calculator. Each step crosses into Python once and synchronises with the
device once. That cost is measured and gated (09).

Out of scope: an integrator on the GPU, batches of several systems,
differentiating through a trajectory. Nothing that uses molix.md needs any of
them today. `MD.run` already runs under `no_grad`, and the system is one
`(1,3,3)` cell (`molix/md/driver.py:445`).

## Context — what the code does today

- **A Python force reaches MD as a duck-typed `Potential`.**
  `SubclassPotential` (`molpy/src/ff/potential.rs:380-430`, chosen at
  `take_potential` `:477`) does the following on every call:
  - allocates a new `(N,3)` numpy array;
  - collects the returned forces into a `Vec`;
  - passes only the coordinates, with no cell, pbc, species or rebuild flag;
  - drops the virial (`ForceTerm::Plain`, `crates/molrs/src/md/evaluator.rs:193-198`).
- **An exception raised in `evaluate` is parked, not raised.** It goes into an
  `ErrSlot`, the step goes on with NaN forces, and the error is re-raised only
  after `run` returns (`molpy/src/md.rs:191-199`).
- **The host step allocates on every step**, although `component.rs:15` says
  `launch` must not:
  - `read_*` / `write_nx3` copy whole columns (`columns.rs:50,54,71`);
  - `FfEvaluator::launch` builds `owned`, `rows` and `explicit` (`evaluator.rs:139-153`);
  - `Neighbor::launch` clones the whole `Neighbors` (`neighbor_list.rs:99`);
  - the zero-force array is allocated (`infra.rs:203`);
  - the phase order is rebuilt as a `Vec` (`driver.rs:254`).
- **`scratch.step` starts from 0 again on every `run`** (`driver.rs:214`). Two
  chained runs therefore draw the same Langevin noise (`fix_langevin.rs:120`).
  `FixLangevin` also accepts `t_stop` and ignores it (`fix_langevin.rs:70-77`).
- **The driver has no unit system, but its code is only right in one.** It
  applies `a = f/m` with no conversion factor (`verlet.rs:51`), and the kinetic
  energy is formed with `mvv2e = 1` (`infra.rs:49-50,67-68`). That is correct
  only in a coherent system (energy = mass·length²/time²). With Å and amu,
  `dt = 1.0` therefore means 48.888 fs for kcal/mol and 10.181 fs for eV.
  Wrong statements follow from this in several places:
  - the docs in `md/__init__.py:10` (argon `PairLjCut` in kcal/mol with
    `VelocityVerlet(1.0)`);
  - `component.rs:43` ("fs when the rest of the frame is");
  - `maxwell.rs:16`;
  - `infra.rs:67`, which carries the `real` k_B into a run of any unit system;
  - `virial.rs:5,12,40-42`, which says "kcal/mol" and gives a factor to bar.
  molix avoids all of this by working in amu·Å²/fs² (`molix/units.py:32`).
- **Rebuilds.** `VerletSkin::update` already reports whether it rebuilt
  (`core/neighbors/verlet_skin.rs:382`). But `Neighbor` and `VerletSkin` keep
  the box they were built with (`neighbor_list.rs:87-90`,
  `verlet_skin.rs:120,316`), so after a change of cell the minimum-image
  displacements are stale.
- **`MD.run` holds `borrow_mut` on the frame arena for the whole run**
  (`molpy/src/md.rs:192-196` → `crates/molrs-ffi/src/frame_ref.rs:83-85`). A
  Python object cannot read the frame from inside a step.
- **Virial conventions.**
  - molrs stores `W = Σ f⊗r` in the six-component order
    `[xx,yy,zz,xy,xz,yz]` (`virial.rs:5-13`), with `P = (2K + tr W)/(3V)`
    (`virial.rs:43`).
  - molix stores `∂E/∂ε = −Σ r⊗f`, with `P = (2K − tr W)/(3V)`
    (`molix/md/barostat.py:21,65`).
- **What molix.md has that molpy.md lacks:**
  - the Berendsen barostat (`barostat.py`);
  - the trajectory, checkpoint and energy-drift hooks (`runner.py:253,398,492`).

  `VelocityInitHook` is only `MaxwellBoltzmann` called where the run is set
  up; molpy already has `MaxwellBoltzmann`.
- **Who uses molix.md:**
  - `molnex/scripts/matpes_port/run_nve.py`;
  - `molnex/scripts/polar_port/run_npt.py`;
  - `molpot`, through `molix.md.NeighborList` writing `edges.shifts`
    (`molpot/forcefield/potentials.py:1843-1960`).

## Decisions

**D1 — The seam is a new md trait, `molrs::md::Calculator`.** It does not
extend `Potential`, and it is not a new `ForceTerm` variant.
- `Potential::evaluate(&self, coords, acc)` (`ff/potential/mod.rs:277`) has no
  error channel and is shared with Lbfgs.
- The rebuild flag and the step index are MD concepts. Putting them on
  `ForceTerm` would make `ff` depend on the layer above it.
- File: `crates/molrs/src/md/calculator.rs`. Types:
  - `Calculator` (trait): `bind`, `positions_mut`, `compute`, `forces`.
  - `CalculatorStep { step: u64, simbox: Option<&SimBox>, cell_changed: bool,
    rebuild: bool, want_virial: bool }`.
  - `CalculatorResult { energy: F, virial: Option<Virial> }`.
  - `MD::set_calculator` mounts it under a new singular `Kind::Calculator`.
    It may sit beside an `FfEvaluator`; the two sum their forces.
- The force / energy / virial add-onto-frame rule (`evaluator.rs:84-116`)
  becomes one function in `columns.rs`, because it now has two callers.
- molrs-ffi is unchanged: no handle, no capsule, no ABI.

**D2 — The calculator supplies the exchange buffers.** molrs-ffi and molpy
never allocate pinned memory.
- At bind, the calculator hands over `positions` and `forces`, each a `(N,3)`
  float64 C-contiguous writable array. It also hands over `virial`, a
  `(6,)` array, when one is asked for.
- In molnex these are numpy views of pinned torch tensors.
- The core gathers x/y/z straight into `positions_mut()` and scatters
  `forces()` straight onto the columns. One gather and one scatter per step,
  with no intermediate copy.
- The binding holds the arrays as Python objects kept alive by Python, so it
  hands out no view into core storage, which satisfies the ABI rule.

**D3 — Binding happens before the borrow.** The binding's `MD.run` calls
`calculator.bind(frame, skin)` before it enters `with_mut`:
- on the first run, and again after any setter;
- with `frame` the `mp.core.Frame` and `skin` the run's skin.

The calculator reads species, charges and anything else it needs there. Each
step it receives only buffers and flags, never the frame.
`Component::build` checks that N is unchanged.

**D4 — One Python call per step.** The call is
`compute(cell, cell_changed, rebuild, step) -> float`.
- `cell` is a `(3,3)` view that stays valid across steps (row-vector lattice,
  Å), or `None`.
- Forces and the virial are written into the bound buffers; only the energy
  is returned.

The intended molnex body is:
1. an H2D copy `pos_dev.copy_(pos_pinned, non_blocking=True)`, plus the cell
   when it changed;
2. a CUDA-graph replay, with edge vectors `x_j − x_i + S·H` formed in fp64
   and cast to the model dtype on the device (not the absolute positions:
   fp32 spacing at 35 Å is 3.8e-6 Å);
3. D2H copies of the forces, energy and virial into pinned memory;
4. a single `event.synchronize()`.

Per-atom energies are summed in fp64, and forces and the virial are cast up
to fp64 before the kick.

**D5 — Rebuild cadence without a CPU search.** The displacement gate moves
out of `VerletSkin` into a new type, `core::neighbors::VerletCadence`. It
carries `x_hold`, `ago`, `every`/`delay`/`check`, the half-skin bound,
`ndanger` and the restore-progress state (`verlet_skin.rs:113-145,352-430`).
- `VerletSkin` holds a `VerletCadence`, so the gate logic exists once.
- The cadence reads the live box on every update.
- Its criterion covers a change of cell. With `M = H·H_hold⁻¹` and the
  non-affine displacement `Δx̃_i = x_i − M x_i,hold`, it rebuilds when
  `2·max_i|Δx̃_i| + ‖M − I‖₂·r_build ≥ skin`.
  - With a fixed cell this reduces to the present half-skin test.
  - Under Berendsen it lets small strains build up instead of forcing a
    rebuild every step.
- `Scratch` gains a field `rebuild: bool`. It is published at `neighbor`,
  either by `Neighbor` or by a cadence-only component that `set_calculator`
  mounts when no list is mounted. `build` refuses two cadences in one run.

**D6 — The run carries a unit preset.**
- `MD` is constructed with a `UnitPreset` (`real` or `metal`, from
  `core::units`).
- The integrator applies `ftm2v`; the kinetic sum applies `mvv2e`.
  `FixLangevin` and `MaxwellBoltzmann` take k_B and T from the preset, and the
  `kb` argument is deleted.
- `dt` is in the preset's time unit: fs for `real`, ps for `metal`, as in
  LAMMPS.
- A calculator declares the preset its numbers are in. The seam converts once
  (a scalar on the energy, a scalar on the forces and virial) and refuses a
  calculator that declares nothing.
- The virial is documented as being in the run's energy unit, and the
  bar-conversion factor becomes a property of the preset.

This is the one breaking change outside the seam. Every caller changes in
the same change, with no shim.

**D7 — Errors abort at the failing step.**
- New variant: `MdError::External(Box<dyn Error + Send + Sync>)`, carrying the
  step index.
- The binding converts the `PyErr` into it and turns it back into the original
  exception in `PyMD.run`.
- `launch_phase` already stops at the first `Err` (`driver.rs:266-268`).
- The frame is left at mid-step k: positions drifted, velocities half-kicked.
  The docs say so; a checkpoint is the way to recover.
- `ErrSlot` leaves `molpy/src/md.rs`.

**D8 — Python forces reach MD only through `set_calculator`.**
- The duck-typed `evaluate` branch is removed from `MD.set_evaluator`. It
  keeps `PairLjCut`, `Potentials` and `WeightedTerms`.
- The duck-typed `Potential` stays for `Potentials.evaluate` and Lbfgs (their
  NaN-parking is follow-up F2).
- The protocol is `molpy.md.Calculator`, a `typing.Protocol` in
  `molpy/python/molpy/md/calculator.py`, mirroring `ff/potential/protocol.py`.
  It has a `units` attribute and the methods `bind` and `compute`.

**D9 — The barostat follows Berendsen 1984 exactly**, i.e. the cube root, not
the linear form molix uses.
- `md::BerendsenBarostat` runs at `end_of_step`, after `KineticEnergy`
  (order above −100).
- It forms `P` from `v(t+Δt)` and `W(x(t+Δt))`.
- It sets `μ³ = 1 − (βΔt/τ_p)(P0 − P)` and refuses `μ³ ≤ 0`.
- It applies `H' = μH` and `x' = o + μ(x − o)`. Fractional coordinates and
  image flags are unchanged; velocities are not scaled.
- It sets `cell_changed` for the next step.
- `meta["virial"]` missing is a named error, never zero.
- Isotropic first. Anisotropic coupling (`μ_αα` from `P_αα`) is a later link.
- The calculator contract states the molrs convention, `W = Σ d_e⊗f_e`
  (equivalently `−sym ∂E/∂ε` at ε = 0), in molrs order. The molnex adapter
  negates its `∂E/∂ε` and reorders it; ASE Voigt order
  `[xx,yy,zz,yz,xz,xy]` is not accepted.

**D10 — Each observer is a component or a compute, by what it is.**
- `md::Dump` (`end_of_step`, every n steps) writes through `io::FrameWriter`
  (`io/writer.rs:22`; `Writer::new` gains `where Self: Sized`).
- `md::Checkpoint` writes mrec: the frame, the absolute step and the cadence
  state.
- Energy drift is a reduction over `meta`, so it is
  `compute::EnergyDrift`: an accumulator with `push(t, e_total)` and
  `finish()`, following the compute-redesign contract.
  - It reports `slope/N` (energy·atom⁻¹·time⁻¹, as GROMACS does) and the rms
    fluctuation per atom.
  - It does not normalise by `|e0|` (molix `runner.py:604-614`), which lets a
    MatPES gate of 1e-3 pass 5 meV/atom.

## As built (01–05, 2026-10-29)

Operator ruling: when maintainability or calculator extensibility conflict
with performance, the first two win. Where the code departs from the
decisions above, it does so for that reason.

- **D1/D2.** The core trait does not lend buffers. It is
  `Calculator::compute(&mut self, step, positions: &[F], forces: &mut [F])`,
  and `bind(&mut self, frame: &Frame, skin)` is called from `Component::build`.
  The Python binding (`molpy/src/md/calculator.rs`) copies positions into the
  buffers the calculator returned from `bind` and copies forces out of them.
  That is one host memcpy each way per step, and the buffers stay the
  calculator's own pinned memory.
- **D3.** Python's `bind` runs inside `build`. It receives a deep-copied
  `Frame` snapshot (`PyFrame::from_core_frame`), so it never touches the
  borrowed arena.
- **D4.** The per-step call is `compute(step, rebuild, box) -> energy`. `box`
  is an `mp.core.Box`, sent only when the cell changed (and on the first call
  after a bind); otherwise it is `None`.
- **D5.** `core::neighbors::VerletCadence` holds the gate, and `VerletSkin`
  delegates to it. The cadence lives inside the mounted calculator component
  (`md::calculator::Mounted`), not on `Scratch`, so two cadences never need
  coordinating. `VerletSkin::set_simbox` / `pairs_into` let the `Neighbor`
  component follow a changing cell and publish into a table that `Scratch`
  owns and reuses.
- **D6.** `MD::new(UnitPreset) -> Result`. The step constants live on the
  preset, `UnitPreset::{ftm2v, mvv2e, factor_from}`, and `KineticEnergy::new`
  derives its `mvv2e` there too. That leaves one formula and no `md::Units`;
  `Component::build` takes `&UnitPreset`. Reduced `lj` units are refused until
  σ, ε and m are defined in the registry.
- **D7.** `MdError::Calculator { name, step, source }`. The binding downcasts
  `source` back to the original `PyErr`.
- **D8.** `MD.set_evaluator` refuses an object that only has `evaluate` and
  points to `set_calculator`. Python pair styles inside `WeightedTerms` still
  park their exceptions in an `ErrSlot`, because `Potential` has no error
  channel. That is F2.
- **01.**
  - Langevin decision: `FixLangevin(temperature, damp, dt, seed)`. The ignored
    `t_stop` is gone; a ramp is a later feature.
  - The step counter is absolute: the setup evaluation is step 0, and steps
    are 1..n across runs.
  - The phase schedule is built once in `build`.
  - Column writes happen in place.
  - The pair table is no longer cloned every step.
  - Per-atom gathers still allocate, and `Component`'s contract now says so.
  - There is no allocation-counting test: AGENTS.md tests check correctness
    only, and step cost is gate 09's job.
- `Writer::new`'s `Self: Sized` is left to link 07, its first user.

Fixed during the post-implementation reviews (architecture, bindings,
science):
- The Langevin noise key XOR-ed the atom id with the channel, so atom 0's y
  draw was atom 2's x draw. Each field is now hashed in its own stage.
  `atoms_and_channels_draw_distinct_noise` covers it.
- A `Potentials` holding a Python `evaluate` was a way around D8. MD now
  refuses it by name.
- A calculator reading the frame it steps panicked on the `RefCell` borrow.
  `FrameRef` / `BlockRef` now use `try_borrow`, so the access fails with
  `FfiError::FrameBusy`: `RuntimeError` in Python, `MOLC_STATUS_FRAME_BUSY` in
  C. Along the way, `Frame.__getitem__` had been swallowing every FFI error as
  a `KeyError`; it now passes them through.
- A buffer resized in place, buffers that share memory, and a non-finite
  energy or force are each refused by name.
- `MD`'s components are no longer taken out of their slot to launch, so a
  panic cannot leave the driver without one.

Deferred:
- The md manual page (`docs/`) waits for 06–07, so the page shows NVT/NPT
  and output once.
- Link 07's checkpoint must reach the cadence inside `Mounted` and the step
  counter (`MD::step` is read-only today).
- `molpy/stubtest-allowlist.txt` exempts `molpy._core.md` as a whole, so
  stubtest does not check the md signatures. They were checked by hand on
  2026-10-29.
- molwasm, molc and molcxx do not bind `md`.

## Chain

### 01-step-hygiene

- **Goal:** `launch` stops allocating on every step, and the step counter is
  absolute.
- **Scope:**
  - Columns are written in place.
  - The evaluator buffers live on the struct.
  - The neighbour pairs are reused with `clone_from`.
  - The phase schedule is precomputed in `build`.
  - `MD` owns a `u64` step that carries across `run` calls and
    `Checkpoint`.
  - `FixLangevin` either ramps `t_start → t_stop` over the run or refuses
    `t_start ≠ t_stop`; the decision is made in the link.
- **Acceptance** (`cargo mrs-test -- md`):
  - (a) A test under a counting global allocator runs an LJ pair + Langevin
    `MD` and asserts zero allocations in steps 2..n. This test proves the
    rule in `component.rs:15`.
  - (b) `run(f,5); run(f,5)` and `run(g,10)` with the same seed give
    bit-identical frames.
  - (c) The `t_stop` behaviour is checked against a hand-written number.
- **Risks:** `Block::insert` may not support in-place writes into an existing
  column. If it cannot, the column API gains an in-place writer in the same
  link.

### 02-units

- **Goal:** D6.
- **Scope:**
  - `MD::new(units)` / `md.MD(units="metal")`.
  - `ftm2v` and `mvv2e` are applied.
  - `kb` is deleted from `FixLangevin`, and `MaxwellBoltzmann` takes T.
  - The docs listed under Context are fixed, including the
    `md/__init__.py` example.
  - `.pyi` and the manual change in the same commit.
- **Acceptance:**
  - (a) One VV step in `real` matches a hand number: m = 12, f = 1
    kcal/mol/Å, dt = 1 fs, so Δv = ½·dt·f/m·ftm2v with ftm2v = 4.184e-4.
  - (b) The same step in `metal`, with ftm2v = 1/1.0364269e-4.
  - (c) The 1-D VV oscillator (m = k = 1, h = 0.1, q0 = 1, p0 = 0) gives
    q = 0.995, p = −0.09975 exactly after one step. `H̃ = ½p²/(1−h²/4) + ½q²`
    stays at 0.5 to rounding over 10⁴ steps.
  - (d) `scripts/check.sh test python docs` passes.
- **Risks:** this touches every MD caller and doc block. molnex is not
  affected, because it does not call molpy.md yet.

### 03-cadence

- **Goal:** D5.
- **Scope:**
  - `VerletCadence`, used by `VerletSkin`.
  - The live box in `Neighbor`.
  - `Scratch.rebuild`.
  - A cadence-only component.
- **Acceptance:**
  - (a) Hand cases for the fixed-cell criterion: a displacement of 0.24·skin
    does not rebuild, 0.26·skin does.
  - (b) An isotropic strain with `‖M−I‖·r_build = 0.9·skin` and no non-affine
    motion does not rebuild; 1.1·skin does.
  - (c) After a change of cell, the pair kernel sees the new box. A test
    compares LJ energy against a hand-computed value in the scaled box.
  - (d) Two cadences in one run are refused.

### 04-calculator-core

- **Goal:** D1, D7, and the core side of D6 (declared units, conversion).
- **Scope:**
  - `md::Calculator`, `CalculatorStep`, `CalculatorResult`.
  - `MD::set_calculator`.
  - `MdError::External`.
  - The shared helper for adding results onto the frame.
- **Acceptance** (Rust, with a harmonic-bond calculator written in the test):
  - (a) Forces and energy match the `FfEvaluator` bond of `evaluator.rs:307`
    on the same frame.
  - (b) The virial passes through, and its absence removes `meta["virial"]`.
  - (c) An error raised at step k aborts `run` with k in the error.
  - (d) A calculator in `metal` driving a `real` run is converted by the
    exact preset ratio.
  - (e) Zero allocations per step, using the allocator from 01.

### 05-calculator-binding

- **Goal:** D2, D3, D4, D8 in molpy.
- **Scope:**
  - `molpy/src/md/calculator.rs` (`PyCalculator`).
  - `molpy.md.Calculator` (Protocol).
  - `.pyi` and the manual page.
  - The duck-typed branch of `MD.set_evaluator` is deleted.
  - `ErrSlot` is removed from `md.rs`.
- **Acceptance** (`scripts/check.sh python`, `tests/test_md/`, numpy only):
  - (a) A harmonic calculator written in numpy runs NVE. Its `positions`
    buffer at each `compute` equals the frame's x/y/z.
  - (b) The data pointer of every bound buffer, from
    `__array_interface__['data']`, is the same on every step. That shows the
    buffers are reused, not reallocated.
  - (c) `bind` is called once for two chained `run`s, and again after a
    setter.
  - (d) An exception at step 3 surfaces from `run` as the same exception
    type, and `compute` was called 3 times after setup.
  - (e) The `rebuild` and `cell_changed` flags arrive as the cadence decided.
  - (f) A wrong buffer is refused at bind with a named error: wrong shape,
    wrong dtype, not contiguous, or read-only.
  - (g) `set_evaluator(object_with_evaluate)` raises `TypeError` and names
    `set_calculator`.

### 06-barostat

- **Goal:** D9.
- **Acceptance:**
  - (a) Ideal gas: 3 atoms, m = 2, with unit velocities scaled by 10 along x,
    y and z, so K = 300.
    - L = 10, W = 0, P0 = 0, β = 0.5, Δt/τ = 0.1.
    - Then μ³ = 1.01, V' = 1010, L' = 10.033223.
    - An atom at x = 3 with o = (−5,−5,−5) maps to 3.026578.
  - (b) The `evaluator.rs:307` bond with v = 0 gives W_xx = 48 and
    P = 0.016. With β = 1 and Δt/τ = 0.5, V' = 1008.
  - (c) A missing virial is refused, and μ³ ≤ 0 is refused.
  - (d) The next step reports `cell_changed`.
- **Risks:** the pressure units come from the preset (02).

### 07-dump-checkpoint

- **Goal:** D10, writers.
- **Acceptance:**
  - (a) `Dump` every 3 steps over 10 steps writes 4 frames (steps 0, 3, 6, 9)
    into a `bytearray` sink, whose coordinates equal the frame at those
    steps.
  - (b) `run 5 → Checkpoint → restore → run 5` is bit-identical to `run 10`,
    with Langevin and a rebuild inside the window.

### 08-energy-drift

- **Goal:** `compute::EnergyDrift`.
- **Acceptance:**
  - (a) E = (δ, −δ, −δ, δ) at t = 0..3 gives slope 0 and rms δ.
  - (b) E = −100 + 0.002·t with N = 4 gives 5e-4 per atom per unit time.
  - (c) The VV oscillator of 02(c), fed through it, gives |slope| < 1e-12.

### 09-gate (out-of-source; never committed, per AGENTS.md Testing)

- **Setup:** GH200, MatPES MACE (official weights) through the molnex
  adapter, against `molix.md` on the same model, at 193 / 1 544 / 4 425
  atoms, fp64 and fp32.
- **Pass:**
  - (a) `t_step(molpy.md) ≤ t_step(molix.md)` at every size and dtype.
  - (b) The overhead outside the model, `t_step − t_op`, is ≤ 0.15 ms at
    every size. `t_op` is the CUDA-graph replay alone, which the notes put at
    0.56 ms for 256 atoms.
  - (c) NVE at 0.5 fs, fp64: |drift| ≤ 0.05 meV/atom/ps (the scale of the
    GROMACS default `verlet-buffer-tolerance`). fp32 drift ≤ 2× fp64 + 0.05.
    The rms fluctuation ratio between 1 fs and 0.5 fs is ≈ 4.
  - (d) An NPT run of the Polar PEO box with Berendsen stays stable for 10⁴
    steps.
- **Record:** the numbers go into molnex notes, and a short summary into this
  spec before it is deleted.

### Downstream (molnex, own spec)

- The adapter implements `molpy.md.Calculator` with pinned buffers, the GPU
  `molix.neighbor.NeighborList`, the CUDA-graph replay contract and one sync.
- Its virial is `−∂E/∂ε`, reordered to molrs order.
- `molpot`'s MD path takes `edges.shifts` from the new list
  (`potentials.py:1843-1960`).
- `run_nve.py` and `run_npt.py` are ported.
- `molix.md`, `get_neighbor_pairs` and `molix.F.locality` are deleted (the
  neighbor-01 "migration first" branch).
- The molnex-side defects in its barostat are moot once molix.md is deleted:
  - the barostat Δt is applied once per chunk (`runner.py:214-215`);
  - P is formed from the pre-scale virial and the post-scale volume
    (`runner.py:220-226`);
  - the `clamp(min=1e-6)` hides divergence (`barostat.py:130`);
  - the sign is stated wrongly at `notes.md:485`.

## Found (iron law)

The architecture and science reviews of 2026-10-29 found these:

| Finding | Where | Severity | Fixed in |
|---|---|---|---|
| The time unit of `dt` is implicit (48.9 / 10.2 fs); the manual example is wrong | `verlet.rs:51`, `md/__init__.py:10`, `component.rs:43`, `maxwell.rs:16` | critical | 02 |
| `real` k_B is carried into any unit system | `infra.rs:67` | high | 02 |
| Allocation every step, against the `launch` contract | `columns.rs:50,71`, `evaluator.rs:139-153`, `neighbor_list.rs:99`, `infra.rs:203`, `driver.rs:254` | high | 01 |
| The step counter restarts every `run`, so Langevin noise repeats | `driver.rs:214` | high | 01 |
| A Python force error is parked as NaN until `run` returns | `molpy/src/md.rs:197`, `potential.rs:417-422` | high | 05 (MD); F2 (Lbfgs) |
| The box is frozen at build, so the skin goes stale under a cell change | `neighbor_list.rs:87-90`, `verlet_skin.rs:120,316` | high (blocks 06) | 03 |
| `t_stop` is accepted and ignored | `fix_langevin.rs:70-77` | medium | 01 |
| The virial docs say kcal/mol / bar for any unit system | `virial.rs:5,12,40-42` | medium | 02 |
| `Writer::new` prevents a writer being used as a trait object | `io/writer.rs:16` | low | 07 |

Follow-ups outside this chain:
- F1: anisotropic Berendsen, and a barostat that conserves energy
  (MTK / Nosé–Hoover), when someone needs one.
- F2: give `Potential` an error channel, so Lbfgs and `Potentials.evaluate`
  stop parking NaN.

## Open questions

1. **Pinned staging vs mapped host memory on GH200.** The baseline is a
   `non_blocking` copy into device memory. The alternative is memory
   registered with `cudaHostRegister`, or plain system memory read directly
   by the graph over NVLink-C2C (ATS). The choice only affects the molnex
   adapter, not the seam. 09 measures both and keeps the faster one.
2. ~~The `t_stop` ruling in 01~~ — closed: one temperature; a ramp is a
   later feature.
