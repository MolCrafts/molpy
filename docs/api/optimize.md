# Optimization

Geometry optimization using potential energy functions.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `LBFGS` | Limited-memory BFGS optimizer | Geometry relaxation of small/medium structures |
| `BondPotentialWrapper` | Adapts bond potentials to Frame interface | Optimizer integration |
| `AnglePotentialWrapper` | Adapts angle potentials to Frame interface | Optimizer integration |

## Related

- [Potential](potential.md) -- energy/force implementations used by optimizers

---

## Full API

### Base

::: molpy.optimize.base

### LBFGS

::: molpy.optimize.lbfgs

### Potential Wrappers

::: molpy.optimize.potential_wrappers
