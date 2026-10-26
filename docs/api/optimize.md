# Optimization

Geometry optimization with the native L-BFGS minimizer.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Lbfgs` | Limited-memory BFGS over the `Potentials` compiled for a frame | Geometry relaxation of small/medium structures |
| `OptimizationReport` | Outcome record: `converged`, `final_energy`, `final_fmax`, `final_grad_rms`, `n_steps` | Inspecting why a run stopped |

Both live in `mp.optimize`; see the
[user guide](../user-guide/08_geometry_optimization.md) for the composition
(typify → `PotentialCompiler(ff).compile(frame)` → `minimize`).

## Related

- [Potential](potential.md) -- energy/force implementations the optimizer drives

---

## Full API

::: molpy.optimize.Lbfgs

::: molpy.optimize.OptimizationReport
