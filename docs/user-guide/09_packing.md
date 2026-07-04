# Packing Systems into a Box

Fill a simulation cell with hundreds of molecules under geometric constraints —
from Python, through the **Packmol** backend.

!!! note "Prerequisites"
    Packing shells out to the **Packmol** executable. Install it and make sure
    `packmol` is on your `PATH` (or pass an explicit path to `Packmol(...)`).
    Without it the packer cannot run.

## What packing solves

Building one molecule gives you one molecule. A simulation needs a *box* — often
hundreds or thousands of molecules arranged without steric clashes. Doing that by
hand (a grid, random insertion) either wastes volume or produces overlaps.

**`Molpack` collects packing *targets* (a molecule + how many + where it may go)
and asks Packmol to place them without clashes, returning a single packed
`Frame`.**

## Packing a box

```python
from molpy.pack import Molpack, InsideBoxConstraint

p = Molpack("pack_out")                      # workdir for Packmol's scratch files
p.add_target(
    water,                                   # a molrs Frame (one molecule)
    number=500,                              # how many copies
    constraint=InsideBoxConstraint(length=30.0),   # keep them inside a 30 Å cube
)
packed = p.optimize(max_steps=1000, seed=42)  # -> a single packed Frame
```

`optimize` returns the packed `Frame` directly. Add several `add_target` calls
before `optimize` to pack a mixture (e.g. solute + solvent) in one run.

## The pieces

| Object | Role |
|---|---|
| `Molpack(workdir)` | The packing session. `workdir` holds Packmol's input/output scratch files. |
| `add_target(frame, number, constraint)` | Register `number` copies of `frame`, restricted by `constraint`. Returns the `Target`. |
| `optimize(max_steps=1000, seed=None, pbc=None)` | Run Packmol and return the packed `Frame`. `pbc` supplies a periodic cell for minimum-image spacing. |
| `Packmol(executable=None, workdir=None)` / `get_packer()` | The backend wrapper, if you need to point at a specific Packmol binary. |

### Constraint catalog

A constraint restricts *where* a target's copies may be placed. Combine them with
`AndConstraint` / `OrConstraint`.

| Constraint | Keeps molecules… |
|---|---|
| `InsideBoxConstraint(length, origin=(0,0,0))` | inside an axis-aligned cube of edge `length` at `origin`. |
| `OutsideBoxConstraint(origin, lengths)` | outside a box (carve out a cavity). |
| `InsideSphereConstraint(radius, center)` | inside a sphere. |
| `OutsideSphereConstraint(radius, center)` | outside a sphere. |
| `MinDistanceConstraint(dmin)` | at least `dmin` apart (clash avoidance). |
| `AndConstraint(a, b)` / `OrConstraint(a, b)` | satisfying both / either sub-constraint. |

```python
from molpy.pack import InsideSphereConstraint, MinDistanceConstraint, AndConstraint

# inside a 20 Å sphere AND no closer than 2.5 Å to each other
c = AndConstraint(
    InsideSphereConstraint(radius=20.0, center=(0.0, 0.0, 0.0)),
    MinDistanceConstraint(dmin=2.5),
)
```

## Parameters that matter

- **`number`** — copies per target. Total atom count = Σ(number × atoms/molecule);
  this drives both memory and Packmol runtime.
- **`max_steps`** — Packmol's placement effort. Raise it for dense boxes that
  fail to converge; lower it for quick drafts.
- **`seed`** — reproducible packings. Pin it to get the same box twice.
- **box size vs `number`** — too many molecules for the volume makes Packmol
  struggle or fail; leave head-room, or pack in stages.

## Pitfalls

- **`packmol` not found** → the run fails immediately. Install it / pass the
  path to `Packmol(executable=...)`.
- **Over-dense boxes** don't converge. If `optimize` stalls, either enlarge the
  box constraint or reduce `number`.
- The returned object is a plain `Frame`; attach a box / periodicity yourself if
  the downstream step needs one.

## See also

- [Building Polymers](02_polymer_stepwise.md) — producing the molecules you pack.
- [Polydisperse Systems](05_polydisperse_systems.md) — packing a chain-length
  distribution.
- [API Reference — Packing](../api/pack.md) — full class/constraint reference.
