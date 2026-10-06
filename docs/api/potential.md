# Potential

Numerical potential energy functions for bonds, angles, dihedrals, and pairs,
and the force-field IR they are declared in.

## Quick reference

The numerical kernels live in the native core. A force field names them
through its styles (`ff.def_style(kind, name)`); `PotentialCompiler` binds them
to a typed `Frame` as a `Potentials` evaluator. There is no Python-side
potential class per style. `mp.potential` re-exports, by identity, molrs's
`molrs.ff.potential` (`Potential`, `kernel`, `LJCut`) and the essentials of the
force-field IR `molrs.ff.ir` (`StyleSpec`, `Param`, `register_style`,
`register_category`, `styles`, `IrError`, …): a new style or category is
registered from Python with nothing rebuilt — see
[Extending the Force Field](../developer/extending-forcefield.md).

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.ForceField` | Styles and types by category; `def_style(kind, name)` | Declaring parameters |
| `mp.BondStyle` / `mp.AngleStyle` / `mp.DihedralStyle` / `mp.ImproperStyle` / `mp.PairStyle` | One kernel name per style, `def_type(...)` for its parameters | Bonded and nonbonded terms |
| `mp.PotentialCompiler` | `PotentialCompiler(ff).compile(frame)` → `Potentials` | Binding a force field to a typed frame |
| `mp.Potentials` | `calc_energy(frame)` / `calc_forces(frame)`; `push` moves more members in | Energy / force computation |
| `mp.potential.kernel` | `kernel(category, style, atoms, **params)` → `Potentials`: any registered style over explicit instances | Assembling terms by hand |
| `mp.potential.LJCut` | The one-type `lj/cut` kernel an MD integrator feeds from a neighbour list | `mp.md` integrators |
| `mp.potential.Potential` | The protocol: `calc_energy_forces(pos) -> (energy, forces)` | Custom (NN / external) forces in MD |
| `mp.potential.StyleSpec`, `Param`, `register_style`, `register_category` | Declare a style (expression or Python kernel) or a category | Extending the force field |

## Canonical example

Define styles and types on a `ForceField`, then compile it against a typed
`Frame` with `mp.PotentialCompiler(ff).compile(frame)`. There is no
per-style `to_potential()`, no `ff.to_potentials()`, and no parameter-array
lookup; the math runs in the high-performance backend.

```python
import molpy as mp

ff = mp.ForceField(name="demo", units="real")
atom_style = ff.def_style("atom", "full")
ct = atom_style.def_type("CT", mass=12.011, charge=-0.18, element="C")
hc = atom_style.def_type("HC", mass=1.008, charge=0.06, element="H")
# a type is given its name, then its endpoints; param name is "k", not "k0"
ff.def_style("bond", "harmonic").def_type("CT-HC", ct, hc, k=340.0, r0=1.09)

# A typed frame: an atoms block + a bonds block carrying a "type" column.
frame = mp.Frame(
    blocks={
        "atoms": {"x": [0.0, 1.2], "y": [0.0, 0.0], "z": [0.0, 0.0], "type": ["CT", "HC"]},
        "bonds": {"atomi": [0], "atomj": [1], "type": ["CT-HC"]},
    }
)

pots = mp.PotentialCompiler(ff).compile(frame)
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
```

## Related

- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

::: molpy.PotentialCompiler

::: molpy.Potentials

::: molpy.potential
