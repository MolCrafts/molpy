# Notes

Evolving decisions, captured with `/mp-note`. A note is promoted into
AGENTS.md once it is stable, then deleted here. Each entry: id and date,
decision, **Why**, **Applies to**.

### mrec-gate-red-until-04 (2026-10-22)

The `mrec` gate stays red until 04-downstream: the molrec repository's
`tests/molrs_adapter.py` still imports the deleted Python `molrs` package.
The fix belongs in molrec (adapter → `molpy`), not in a shim here.
**Why:** no backward-compatibility apparatus; downstream migrates in 04.
**Applies to:** `scripts/check.sh mrec`, `scripts/partners.py`.

### md-maintainability-over-performance (2026-10-29)

In molpy.md, when maintainability or the extensibility of the calculator
seam conflicts with performance, maintainability and extensibility win. A
host-side cost is accepted when removing it would complicate a contract. For
example, `Calculator::compute` takes `&[F]` / `&mut [F]` and the binding does
one memcpy per direction rather than lending buffers, and per-atom column
gathers still allocate. Step cost is judged by spec gate 09, not by a unit
test.
**Why:** the operator's ruling while fixing the md-calculator-seam findings.
**Applies to:** `crates/molrs/src/md/`, `molpy/src/md/`,
`.claude/specs/md-calculator-seam.md`.

### md-run-units (2026-10-29)

An MD run carries a unit preset: `MD::new(UnitPreset)` / `mp.md.MD(units)`,
with no default. `dt` is in the preset's time unit (fs in `real`, ps in
`metal`, as in LAMMPS). The step applies `UnitPreset::ftm2v` and `mvv2e`.
Thermostats and velocity draws take a temperature and read k_B from the
preset. There is no unit-agnostic MD and no `kb` / `kbt` argument. Reduced
`lj` units are refused until σ, ε and m are defined in the registry.
**Why:** `a = f/m` with no factor silently made `dt = 1.0` mean 48.9 fs in
`real`.
**Applies to:** `crates/molrs/src/md/`, `crates/molrs/src/core/units/preset.rs`,
`molpy/src/md/`, `molpy/python/molpy/md/`.

### md-force-providers (2026-10-29)

MD takes a force provider of its own (an ML potential, an external program)
only as a `molpy.md.Calculator` (`MD.set_calculator`). The calculator supplies
its exchange buffers at `bind` and fills them in place. The driver owns the
rebuild cadence (`VerletCadence`) and tells it `rebuild`. A failure stops the
run at its step with the original exception. `MD.set_evaluator` refuses
duck-typed `evaluate` objects, including ones inside `Potentials`. Those stay
for `Potentials.evaluate` and `Lbfgs`.
**Why:** one seam for torch models, with cell, cadence and error path, and no
NaN-parking in MD.
**Applies to:** `crates/molrs/src/md/calculator.rs`, `molpy/src/md/`,
`molpy/src/ff/potential.rs`.
