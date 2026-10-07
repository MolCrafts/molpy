# Geometry Optimization

Take the strain out of a freshly built structure: `LBFGS` relaxes it to a
local force-field minimum and reports why it stopped.

## When you need it

A freshly built or packed structure is rarely at a force-field minimum: bond
lengths, angles, and close contacts carry excess energy. Before a production
simulation — or to compare energies meaningfully — you minimize the geometry so
the forces drop below a tolerance.

**`LBFGS` moves atoms downhill on a set of potentials until the maximum force
falls under `fmax`.** The minimizer is the native limited-memory quasi-Newton
implementation, re-exported as `molpy.LBFGS`; it
drives the `Potentials` that `PotentialCompiler` compiles from a force field for your frame.

## Minimizing a structure

```python
import molpy as mp

mol, _ = mp.Conformer(seed=42).generate(mp.io.SmilesIR("CCO").to_atomistic())
typifier = mp.ff.typifier.OPLSAATypifier()
frame = typifier.typify(mol).to_frame()
forcefield = typifier.forcefield()  # OPLS-AA parameters of the types just assigned

potentials = mp.ff.potential.PotentialCompiler(forcefield).compile(frame)  # bonded + pair terms
opt = mp.LBFGS(potentials, fmax=0.05, max_steps=200)
frame, report = opt.run(frame)  # a new frame with the relaxed coordinates

print(report.converged, report.final_energy, report.final_fmax, report.n_steps)
```

`run` never mutates its input: it returns the relaxed frame and an
`OptReport`. Keep the returned frame; the one you passed in is unchanged.

## Parameters

`LBFGS(potentials, *, fmax=0.05, max_steps=500, max_step=0.2, memory=8)`:

| Parameter | Effect |
|---|---|
| `fmax` | Convergence threshold on the largest force component (kcal/mol/Å). The run stops when every force is below it. |
| `max_steps` | Hard cap on iterations — a safety net if `fmax` is never reached. |
| `max_step` | Largest atomic displacement per step (Å). Smaller = more stable but slower; raise it only if convergence is sluggish and stable. |
| `memory` | Number of past steps the L-BFGS Hessian approximation keeps. More memory = better curvature estimate, more storage. |

`run` also accepts a bare `(N, 3)` coordinate array (or a `(B, N, 3)` batch)
when you already hold coordinates outside a frame; it returns arrays of the
same shape.

## Reading the result

`OptReport` carries `converged`, `final_energy`, `final_fmax` and `n_steps`.
Always check `converged`: a run that hit `max_steps` (`converged = False`)
has *not* reached the minimum — loosen `fmax`, raise `max_steps`, or inspect
the structure.

## Pitfalls

- **Not converged ≠ minimized.** A `False` `converged` means you stopped at
 the step cap.
- **Units are the native units:** energies in kcal/mol, forces in kcal/mol/Å,
 lengths in Å. A threshold that is too tight for a coarse force field never
 converges; too loose leaves residual strain.
- The potentials are compiled for one topology. Relaxing a frame whose bonds
 or types changed needs `PotentialCompiler(forcefield).compile(frame)` again.
- Optimization needs a *typified* frame with a force field — run a typifier
 first, otherwise `compile` has no types to look up.

## See also

- [Force Field](../tutorials/04_force_field.md) — building the `ForceField` you optimize
 against.
- [3D Conformer Generation](07_conformers.md) — the graph-embedding
 step that precedes force-field relaxation.
- [Engine](12_engine.md) — running full dynamics after minimization.
