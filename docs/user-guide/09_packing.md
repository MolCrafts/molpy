# Packing Systems into a Box

Fill a simulation cell with hundreds of molecules under geometric constraints —
from Python, through molpy's **`Packmol` wrapper**.

!!! note "Prerequisites"
    `Packmol` is molpy's thin **wrapper** around the
    [Packmol](https://m3g.github.io/packmol/) executable: it writes the input
    deck, shells out to `packmol`, and reads the result back into a `Frame`.
    Install Packmol and put `packmol` on your `PATH` (or point at it explicitly:
    `Packmol(executable="/path/to/packmol")`).

!!! tip "Prefer a pure-Python packer? Try `molcrafts-molpack`"
    [**molcrafts-molpack**](https://molcrafts.github.io/molpack/) is our own
    dependency-free, Python-native packer — a Rust Packmol port with collective
    restraints and **no external binary**. `pip install molcrafts-molpack`; it
    speaks the same molrs `Frame`, so it drops straight into the workflows below.
    Give it a spin.

## What packing solves

Building one molecule gives you one molecule. A simulation needs a *box* — often
hundreds or thousands of molecules arranged without steric clashes. Doing that by
hand (a grid, random insertion) either wastes volume or produces overlaps.

**molpy's `Packmol` wrapper collects packing *targets* (a molecule + how many +
where it may go) and drives the Packmol executable to place them without clashes,
returning a single packed `Frame`.**

## Packing a box

```python
from molpy.pack import Packmol, InsideBoxConstraint

p = Packmol(workdir="pack_out")                      # workdir for Packmol's scratch files
p.def_target(
    water,                                   # a molrs Frame (one molecule)
    number=500,                              # how many copies
    constraint=InsideBoxConstraint(length=30.0),   # keep them inside a 30 Å cube
)
packed = p(max_steps=1000, seed=42)  # -> a single packed Frame
```

Calling the packer — `p(...)` — runs Packmol and returns the packed `Frame`
directly. Register several `def_target`s before that call to pack a mixture
(e.g. solute + solvent) in one run.

## The pieces

| Object | Role |
|---|---|
| `Packmol(executable=None, workdir=None)` | molpy's Packmol **wrapper** — the packing session. `workdir` holds Packmol's scratch files; `executable` points at a specific `packmol` binary. |
| `def_target(frame, number, constraint)` | Register `number` copies of `frame`, restricted by `constraint`. Returns the `Target`. |
| `packer(max_steps=1000, seed=None, pbc=None)` | Call the packer to run Packmol and return the packed `Frame`. `pbc` supplies a periodic cell for minimum-image spacing. |

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
