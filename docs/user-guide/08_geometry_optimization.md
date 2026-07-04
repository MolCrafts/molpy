# Geometry Optimization

Take the strain out of a freshly built structure: `molpy.optimize` relaxes it
to a local force-field minimum and tells you why it stopped.

## When you need it

A freshly built or packed structure is rarely at a force-field minimum: bond
lengths, angles, and close contacts carry excess energy. Before a production
simulation — or to compare energies meaningfully — you minimize the geometry so
the forces drop below a tolerance.

**An `Optimizer` moves atoms downhill on a potential until the maximum force
`fmax` falls under a threshold.** MolPy ships `LBFGS`, a limited-memory
quasi-Newton minimizer, evaluating a force field through `ForceFieldPotential`.

## Minimizing a structure

`molpy.optimize` is imported directly (it is not exposed as `mp.optimize`):

```python
from molpy.optimize import LBFGS, ForceFieldPotential

potential = ForceFieldPotential(forcefield)      # wraps a molrs ForceField
opt = LBFGS(potential)
result = opt.run(frame, fmax=0.05, steps=200)    # relaxes frame in place

print(result.converged, result.energy, result.fmax, result.nsteps)
print(result.reason)                             # why it stopped
```

`run` returns an `OptimizationResult`; by default it optimizes `frame` **in
place** (`inplace=True`). Pass `inplace=False` to keep the input untouched.

## Parameters

`LBFGS(potential, *, maxstep=0.04, memory=20, damping=1.0)`:

| Parameter | Effect |
|---|---|
| `maxstep` | Largest atomic displacement per step (Å). Smaller = more stable but slower; raise it only if convergence is sluggish and stable. |
| `memory` | Number of past steps the L-BFGS Hessian approximation keeps. More memory = better curvature estimate, more storage. |
| `damping` | Scales each proposed step (`1.0` = undamped). Lower it if the optimizer overshoots on stiff systems. |

`run(frame, fmax=0.01, steps=1000, *, inplace=True)`:

| Parameter | Effect |
|---|---|
| `fmax` | Convergence threshold on the max force (eV/Å). The run stops when every force is below it. |
| `steps` | Hard cap on iterations — a safety net if `fmax` is never reached. |

## Reading the result

`OptimizationResult` carries `frame`, `energy`, `fmax`, `nsteps`, `converged`,
and `reason`. Always check `converged`: a run that hit the `steps` cap
(`converged = False`) has *not* reached the minimum — loosen `fmax`, raise
`steps`, or inspect the structure.

To watch progress, attach a callback that fires every `interval` steps:

```python
opt.attach(lambda: print(opt.step(frame)), interval=10)   # (energy, fmax) per call
```

## Pitfalls

- **Not converged ≠ minimized.** A `False` `converged` with `reason` naming the
  step limit means you stopped early.
- **`fmax` units are eV/Å.** A threshold that is too tight for a coarse force
  field never converges; too loose leaves residual strain.
- Optimization needs a *typified* frame with a force field — run a typifier first,
  otherwise `ForceFieldPotential` has nothing to evaluate.

## See also

- [Force Field](../tutorials/04_force_field.md) — building the `ForceField` you optimize
  against.
- [3D Conformer Generation](07_conformers.md) — the graph-embedding
  step that precedes force-field relaxation.
- [Engine](12_engine.md) — running full dynamics after minimization.
