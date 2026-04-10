# Potential

Numerical potential energy functions for bonds, angles, dihedrals, and pairs.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `BondHarmonic` | Harmonic bond: E = k(r - r₀)² | Standard bonded interactions |
| `AngleHarmonic` | Harmonic angle: E = k(θ - θ₀)² | Standard angle terms |
| `LJ126` | Lennard-Jones 12-6 pair | Standard nonbonded interactions |
| `TypeIndexedArray` | Array indexed by type name strings | Vectorized parameter lookup |

## Canonical example

```python
ff = mp.AtomisticForcefield(name="demo", units="real")
bond_style = ff.def_bondstyle("harmonic")
bond_style.def_type(ct, hc, k0=340.0, r0=1.09)

pot = bond_style.to_potential()
print(pot.k["CT-HC"])   # [340.]
print(pot.r0["CT-HC"])  # [1.09]
```

## Related

- [Concepts: Force Field](../tutorials/04_force_field.md)

---

## Full API

### Base

::: molpy.potential.base

### Bond

::: molpy.potential.bond

### Angle

::: molpy.potential.angle

### Dihedral

::: molpy.potential.dihedral

### Pair

::: molpy.potential.pair

### Pair Params

::: molpy.potential.pair_params

### Utils

::: molpy.potential.utils
