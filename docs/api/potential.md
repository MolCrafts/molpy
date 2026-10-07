# Potential

Numerical potential energy functions for bonds, angles, dihedrals, and pairs,
and the force-field IR they are declared in.

## Quick reference

The numerical kernels live in the native core. A force field names them
through its styles (`ff.def_style(kind, name)`); `PotentialCompiler` binds them
to a typed `Frame` as a `Potentials` evaluator. There is no Python-side
potential class per style. Each is molrs's submodule by identity:
`mp.ff.potential` is `molrs.ff.potential` (`Potentials`, `WeightedTerms`,
`PairLjCut`, `Potential`, …), `mp.ff.compile` is `molrs.ff.compile`
(`PotentialCompiler`, `compile_explicit_terms`), `mp.ff.ir` is the force-field
IR's vocabulary `molrs.ff.ir` (`ParamSpec`, `StyleSpec`, `CategorySpec`,
`IrError`, …), and `mp.ff.style_registry` is `molrs.ff.style_registry`
(`StyleDeclaration`, `register_style`, `register_category`, `styles`, …): a new
style or category is registered from Python with nothing rebuilt — see
[Extending the Force Field](../developer/extending-forcefield.md).

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `mp.ff.forcefield.ForceField` | Styles and types by category; `def_style(kind, name)` | Declaring parameters |
| `mp.ff.forcefield.BondStyle` / `mp.ff.forcefield.AngleStyle` / `mp.ff.forcefield.DihedralStyle` / `mp.ff.forcefield.ImproperStyle` / `mp.ff.forcefield.PairStyle` | One kernel name per style, `def_type(...)` for its parameters | Bonded and nonbonded terms |
| `mp.ff.compile.PotentialCompiler` | `PotentialCompiler(ff).compile(frame)` → `Potentials` | Binding a force field to a typed frame |
| `mp.ff.potential.Potentials` | `calc_energy(frame)` / `calc_forces(frame)`; `push` moves more members in | Energy / force computation |
| `mp.ff.compile.compile_explicit_terms` | `compile_explicit_terms(category, style, atoms, **params)` → `Potentials`: any registered style over explicit instances | Assembling terms by hand |
| `mp.ff.potential.PairLjCut` | The one-type `lj/cut` kernel an MD integrator feeds from a neighbour list | `mp.md` integrators |
| `mp.ff.potential.Potential` | The protocol: `calc_energy_forces(pos) -> (energy, forces)` | Custom (NN / external) forces in MD |
| `mp.ff.style_registry.StyleDeclaration`, `register_style`, `register_category`; `mp.ff.ir.ParamSpec` | Declare a style (expression or Python kernel) or a category; `StyleSpec` / `CategorySpec` are the registered records `styles()` / `categories()` return | Extending the force field |

## Canonical example

Define styles and types on a `ForceField`, then compile it against a typed
`Frame` with `mp.ff.compile.PotentialCompiler(ff).compile(frame)`. There is no
per-style `to_potential()`, no `ff.to_potentials()`, and no parameter-array
lookup; the math runs in the high-performance backend.

```python
import molpy as mp

ff = mp.ff.forcefield.ForceField(name="demo", units="real")
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

pots = mp.ff.compile.PotentialCompiler(ff).compile(frame)
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
```

## Related

- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

::: molpy.ff.potential

::: molpy.ff.compile

::: molpy.ff.ir

::: molpy.ff.style_registry
