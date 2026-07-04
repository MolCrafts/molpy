# Extending the Force Field

This page shows how to add custom interaction styles and export formatters.

!!! note "Discuss before you build"
    New interaction styles span two repositories (a Rust kernel in molrs plus the Python style and formatters here). Open a [GitHub issue](https://github.com/MolCrafts/molpy/issues) describing the functional form before implementation; the [Architecture Overview](architecture-overview.md) explains where the pieces live.

## Where the math lives

The force-field model — `ForceField`, the `Style` tree, the `Type` tree, and all
energy/force kernels — is owned by the **molrs** Rust extension. molpy does not
maintain a parallel Python potential layer. There is no `style.to_potential()` and
no Python kernel class; evaluation always goes through

```python
ff.to_potentials().calc_energy(frame)   # and .calc_forces(frame)
```

This changes what "extending the force field" means:

1. **Kernel** — the numerical form (energy + forces) is implemented in molrs
   (`molrs-ff`, Rust) and registered there so `ForceField` can dispatch on the
   style name.
2. **Named Style** — on the Python side you expose a thin `Style` subclass whose
   only job is to pin the style name, so callers can write
   `ff.def_style(BondMorseStyle())` instead of `ff.def_bondstyle("morse")`.
3. **Formatters** — serialize the new style's parameters for each export backend
   (LAMMPS, GROMACS, XML).

If molrs already ships the kernel you need, you only do steps 2 and 3 (and step 2
may already exist). Adding a brand-new functional form requires step 1 first.


## Step 1: add the kernel in molrs

A new functional form (e.g. a Morse bond) is implemented in the `molrs-ff` crate:
write the energy and force expressions, then register the kernel under its style
name so `ForceField::to_potentials` can find it. Rebuild the molrs wheel
(`maturin develop` / `maturin build`) and reinstall it; molpy picks up the new
kernel automatically because it re-exports the molrs hierarchy.

Once registered, the style name is usable directly with the generic helpers:

```python
import molpy as mp

ff = mp.ForceField(name="custom", units="real")
a_style = ff.def_atomstyle("full")
c = a_style.def_type("C", mass=12.011)
o = a_style.def_type("O", mass=15.999)

bond_style = ff.def_bondstyle("morse")          # dispatches to the molrs kernel
bond_style.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)
```


## Step 2: expose a thin named Style

For ergonomics and discoverability, give the kernel a named `Style` class. It
carries **no** kernel and **no** `to_potential()` — it only fixes the style name
via `_name_default`. Add it next to the other specialized styles in
`molpy/core/forcefield.py` (and re-export it from `molpy.potential` if desired).

```python
from molpy.core.forcefield import BondStyle

class BondMorseStyle(BondStyle):
    """Bond ``morse`` style (LAMMPS ``bond_style morse``)."""

    def _name_default(self) -> str:
        return "morse"
```

Types and parameters flow through molrs natively — there is no `def_type()`
override to write. Use the named style with `def_style`:

```python
morse = ff.def_style(BondMorseStyle())
morse.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)
```


## Step 3: register param formatters

Each export backend has a `ForceFieldFormatter` subclass that inherits from the
format's `FieldFormatter` (for data field name mapping) and adds `_param_formatters`
(for Style/Type parameter serialization).

Register your style's param formatter on the appropriate subclass:

```python
from molpy.core.forcefield import BondMorseStyle
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter

def _format_morse_bond(typ) -> list[float]:
    """Format morse bond parameters for LAMMPS: D alpha r0"""
    p = typ.params.kwargs
    return [p["D"], p["alpha"], p["r0"]]

LammpsForceFieldFormatter.register_param_formatter(BondMorseStyle, _format_morse_bond)
```

Repeat for each backend. Registrations are **isolated per subclass** — adding a
formatter to one backend does not affect others. This isolation is enforced by
`__init_subclass__` copying the registry.


## Using the custom interaction

Build the model, then evaluate it against a typed `Frame`:

```python
import molpy as mp
import numpy as np
from molpy.core.forcefield import BondMorseStyle

ff = mp.ForceField(name="custom", units="real")
a_style = ff.def_atomstyle("full")
c = a_style.def_type("C", mass=12.011)
o = a_style.def_type("O", mass=15.999)

morse = ff.def_style(BondMorseStyle())
morse.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)

# Two atoms exactly at r0 → Morse energy is 0.
frame = mp.Frame()
atoms = mp.Block()
atoms.insert("x", np.array([0.0, 1.43]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms
bonds = mp.Block()
bonds.insert("atomi", np.array([0], dtype=np.uint32))
bonds.insert("atomj", np.array([1], dtype=np.uint32))
bonds.insert("type", np.array(["C-O"], dtype=str))
frame["bonds"] = bonds

pots = ff.to_potentials()
print(pots.calc_energy(frame))   # 0.0 at r0
```


## Checklist

- [ ] Kernel implemented and registered in `molrs-ff` (Rust), wheel rebuilt
- [ ] Validate the kernel: energy at equilibrium = 0, monotonic increase away from it
- [ ] Thin named `Style` subclass in `molpy/core/forcefield.py` (only `_name_default`)
- [ ] Register formatters for each writer backend (LAMMPS, GROMACS, XML)
- [ ] Write tests: type creation, `to_potentials().calc_energy(frame)` values, export round-trip
- [ ] Tests in `tests/test_core/test_forcefield.py` and `tests/test_io/test_forcefield/`
