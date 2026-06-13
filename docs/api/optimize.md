# Optimization

Geometry optimization using potential energy functions.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Optimizer` | Base class driving any `calc_energy(frame)` / `calc_forces(frame)` potential | Custom optimizers |
| `OptimizationResult` | Result record (final frame, energy, convergence info) | Inspecting outcomes |
| `LBFGS` | Limited-memory BFGS optimizer | Geometry relaxation of small/medium structures |
| `ForceFieldPotential` | Wraps a molrs `ForceField` (via `ff.to_potentials(frame)`) as an optimizer potential | Optimizing with force-field energies/forces |

## Related

- [Potential](potential.md) -- energy/force implementations used by optimizers

---

## Full API

### Base

::: molpy.optimize.base

### LBFGS

::: molpy.optimize.lbfgs

### ForceField Potential

::: molpy.optimize.forcefield_potential
