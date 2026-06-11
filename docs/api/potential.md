# Potential

Numerical potential energy functions for bonds, angles, dihedrals, and pairs.

## Quick reference

The numerical kernels live in the molrs Rust extension; `molpy.potential` exposes
the thin `Style` classes that name them plus the `Potentials` evaluator.

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `BondHarmonicStyle` | Harmonic bond style: E = k(r - r₀)² | Standard bonded interactions |
| `AngleHarmonicStyle` | Harmonic angle style: E = k(θ - θ₀)² | Standard angle terms |
| `LJ126Style` | Lennard-Jones 12-6 pair style | Standard nonbonded interactions |
| `Potentials` | Deferred evaluator over a typed `Frame` | Energy / force computation |

## Canonical example

Define styles and types on a `ForceField`, then evaluate against a typed `Frame`
via `ff.to_potentials()`. There is no per-style `to_potential()` and no
parameter-array lookup; the math runs in molrs.

```python
import molpy as mp
import numpy as np

ff = mp.ForceField(name="demo", units="real")
astyle = ff.def_atomstyle("full")
ct = astyle.def_type("CT", mass=12.011, charge=-0.18, element="C")
hc = astyle.def_type("HC", mass=1.008,  charge=0.06,  element="H")

bond_style = ff.def_bondstyle("harmonic")
bond_style.def_type(ct, hc, k=340.0, r0=1.09)   # param name is "k", not "k0"

# Build a typed frame (atoms block + bonds block carrying a "type" column).
frame = mp.Frame()
atoms = mp.Block()
atoms.insert("x", np.array([0.0, 1.2]))
atoms.insert("y", np.array([0.0, 0.0]))
atoms.insert("z", np.array([0.0, 0.0]))
frame["atoms"] = atoms
bonds = mp.Block()
bonds.insert("atomi", np.array([0], dtype=np.uint32))
bonds.insert("atomj", np.array([1], dtype=np.uint32))
bonds.insert("type", np.array(["CT-HC"], dtype=str))
frame["bonds"] = bonds

pots = ff.to_potentials()
energy = pots.calc_energy(frame)
forces = pots.calc_forces(frame)
```

## Related

- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Bond

::: molpy.potential.bond

### Angle

::: molpy.potential.angle

### Dihedral

::: molpy.potential.dihedral

### Pair

::: molpy.potential.pair
