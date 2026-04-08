# Extending the Force Field

This page shows how to add custom interaction styles, type schemas, potential kernels, and export formatters.

## The four-step extension chain

Adding a new interaction form (e.g., Morse bonds) requires four components:

1. **Type** — declares which parameters exist (`D`, `alpha`, `r0`)
2. **Style** — constructs types and converts them to a potential
3. **Potential** — implements the numerical kernel (`calc_energy`, `calc_forces`)
4. **Formatters** — serialize parameters for each export backend (LAMMPS, GROMACS, XML)

All four are required for a complete, reusable extension.


## Step 1: define the Type

Subclass the appropriate `Type` base. For bonds, subclass `BondType`. Store the parameters in the parent via `super().__init__(name, **kwargs)`.

```python
from molpy.core.forcefield import AtomType, BondType

class MorseBondType(BondType):
    def __init__(self, name: str, itom: AtomType, jtom: AtomType,
                 D: float, alpha: float, r0: float):
        super().__init__(name, itom, jtom, D=D, alpha=alpha, r0=r0)
```


## Step 2: define the Style

Subclass `BondStyle`. Implement `def_type()` to create your custom type, and `to_potential()` to convert all types into the numerical kernel.

```python
from molpy.core.forcefield import BondStyle

class MorseBondStyle(BondStyle):
    def def_type(self, itom, jtom, D, alpha, r0, name=""):
        if not name:
            name = f"{itom.name}-{jtom.name}"
        typ = MorseBondType(name, itom, jtom, D=D, alpha=alpha, r0=r0)
        self.types.add(typ)
        return typ

    def to_potential(self):
        types = self.get_types(MorseBondType)
        return MorseBondPotential(
            D={t.name: t["D"] for t in types},
            alpha={t.name: t["alpha"] for t in types},
            r0={t.name: t["r0"] for t in types},
        )
```


## Step 3: define the Potential

Subclass the appropriate potential base. Implement `calc_energy` and `calc_forces`. Use `TypeIndexedArray` for vectorized parameter lookup by type name.

```python
import numpy as np
from molpy.potential.bond.base import BondPotential
from molpy.potential.utils import TypeIndexedArray

class MorseBondPotential(BondPotential):
    name = "morse"

    def __init__(self, D, alpha, r0):
        self.D = TypeIndexedArray(D).reshape(-1, 1)
        self.alpha = TypeIndexedArray(alpha).reshape(-1, 1)
        self.r0 = TypeIndexedArray(r0).reshape(-1, 1)

    def calc_energy(self, r, bond_idx, bond_types):
        dr = r[bond_idx[:, 1]] - r[bond_idx[:, 0]]
        dr_norm = np.linalg.norm(dr, axis=1, keepdims=True)
        exp_term = 1.0 - np.exp(
            -self.alpha[bond_types] * (dr_norm - self.r0[bond_types])
        )
        return float(np.sum(self.D[bond_types] * exp_term ** 2))

    def calc_forces(self, r, bond_idx, bond_types):
        # Similar implementation with gradient of Morse potential
        ...
```

**Always validate** the potential against known values (energy at r0 should be zero, energy should increase monotonically away from r0).


## Step 4: register param formatters

Each export backend has a `ForceFieldFormatter` subclass that inherits from the format's `FieldFormatter` (for data field name mapping) and adds `_param_formatters` (for Style/Type parameter serialization).

Register your custom Style's param formatter on the appropriate subclass:

```python
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter

def _format_morse_bond(typ) -> list[float]:
    """Format MorseBondType parameters for LAMMPS: D alpha r0"""
    return [typ.params.kwargs["D"], typ.params.kwargs["alpha"], typ.params.kwargs["r0"]]

LammpsForceFieldFormatter.register_param_formatter(MorseBondStyle, _format_morse_bond)
```

Repeat for each backend. Registrations are **isolated per subclass** — adding a formatter to one backend does not affect others. This isolation is enforced by `__init_subclass__` copying the registry.


## Using the custom interaction

```python
import molpy as mp

ff = mp.AtomisticForcefield(name="custom", units="real")
a_style = ff.def_atomstyle("full")
c = a_style.def_type("C", mass=12.011)
o = a_style.def_type("O", mass=15.999)

morse = ff.def_style(MorseBondStyle("morse"))
morse.def_type(c, o, D=100.0, alpha=1.8, r0=1.43)

pot = morse.to_potential()
print(pot.D["C-O"])   # [100.]
```


## Checklist

- [ ] Custom `Type` subclass with explicit parameters in `__init__`
- [ ] Custom `Style` subclass with `def_type()` and `to_potential()`
- [ ] Custom `Potential` subclass with `calc_energy()` and `calc_forces()`
- [ ] Validate potential: energy at equilibrium = 0, monotonic increase away from it
- [ ] Register formatters for each writer backend (LAMMPS, GROMACS, XML)
- [ ] Write tests: type creation, potential values, export round-trip
- [ ] Tests in `tests/test_potential/` and `tests/test_io/test_forcefield/`
