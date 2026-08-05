# Packing Systems into a Box

Fill a simulation cell with hundreds of molecules under geometric restraints —
from Python, through **[molpack](https://docs.molcrafts.org/molpack/)**
(`pip install molcrafts-molpack`).

!!! note "Prerequisites"
    Packing uses **molpack**, not an external Packmol binary. Install the
    wheel and import it:

    ```bash
    pip install molcrafts-molpack
    ```

    molpack speaks the same `Frame` as molpy, so packed systems drop
    straight into typify / export / engine workflows.

## What packing solves

Building one molecule gives you one molecule. A simulation needs a *box* — often
hundreds or thousands of molecules arranged without steric clashes. Doing that by
hand (a grid, random insertion) either wastes volume or produces overlaps.

**molpack** takes packing *targets* (a molecule frame + how many copies + where
they may go) and returns a single packed, topology-complete `Frame`.

## Packing a box

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
import molpy as mp
from molpack import InsideBoxRestraint, Molpack, Target

water, _ = mp.conformer.Conformer(seed=1).generate(mp.io.read_smiles("O"))
water_frame = water.to_frame() # one molecule, as a Frame

water = (
 Target(water_frame, count=500)
.with_name("water")
.with_restraint(InsideBoxRestraint([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]))
)
packed = Molpack().with_seed(42).pack([water], max_loops=200)
# packed is a Frame (1500 atoms for TIP3P water × 500)
```

Register several `Target`s and pass them together to pack a mixture (solute +
solvent) in one run.

## The pieces

| Object | Role |
|---|---|
| `Target(frame, count)` | One species: template `Frame` + number of copies. Immutable builders: `.with_name`, `.with_restraint`, … |
| `InsideBoxRestraint(min, max)` | Keep atoms inside an axis-aligned box (Å). |
| `Molpack()` | Packer session. Chain `.with_seed`, `.with_tolerance`, `.with_periodic_box`, … |
| `packer.pack([targets], max_loops=…)` | Run packing; returns a packed `Frame`. |
| `packer.pack_with_report([targets], max_loops=…)` | Same, plus `PackResult` diagnostics (`.converged`, `.fdist`, …). |

### Restraint catalog

A restraint restricts *where* a target's copies may be placed. Stack several
with repeated `.with_restraint(...)` calls.

| Restraint | Keeps molecules… |
|---|---|
| `InsideBoxRestraint(min, max, periodic=…)` | inside an axis-aligned box. |
| `InsideSphereRestraint(center, radius)` | inside a sphere. |
| `OutsideSphereRestraint(center, radius)` | outside a sphere. |
| `AbovePlaneRestraint` / `BelowPlaneRestraint` | on one side of a plane. |
| `GaussianPlane` / `GaussianPoint` / … | collective distribution-matching (species-level profiles). |

```python
# docs: skip — optional molcrafts-molpack; not a molpy runtime/doc dep
from molpack import InsideSphereRestraint, Target

# confine to a 20 Å sphere about the origin
c = InsideSphereRestraint([0.0, 0.0, 0.0], 20.0)
target = Target(water_frame, count=100).with_restraint(c)
```

Full restraint reference: [molpack docs — restraints](https://docs.molcrafts.org/molpack/python/guide/restraints/).

## Parameters that matter

- **`count`** — copies per target. Total atom count = Σ(count × atoms/molecule).
- **`max_loops`** — outer packing budget. Raise it for dense boxes that fail to
 converge; lower it for quick drafts.
- **`with_seed(n)`** — reproducible packings.
- **box size vs `count`** — too many molecules for the volume will not converge;
 leave head-room, or pack in stages.
- **`with_periodic_box(...)`** — when the simulation cell is periodic, set PBC
 on the packer so spacing uses minimum-image distances.

## Pitfalls

- **`ModuleNotFoundError: molpack`** → `pip install molcrafts-molpack`.
- **Over-dense boxes** don't converge. Enlarge the box restraint or reduce
 `count`.
- The returned object is a plain `Frame`; set `frame.box` for downstream
 writers/engines if the packer did not already attach one.

## See also

- [molpack documentation](https://docs.molcrafts.org/molpack/) — full Python +
 Rust guide, CLI, Packmol-script parity.
- [Assembly](02_assembly.md) — producing the molecules you pack.
- [Polydisperse Systems](05_polydisperse_systems.md) — packing a chain-length
 distribution.
- [API Reference — Packing](../api/pack.md) — molpy surface + molpack entry points.
