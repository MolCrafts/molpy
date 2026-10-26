# Geometry Optimization

Take the strain out of a freshly built structure: `Lbfgs` relaxes it to a
local force-field minimum and reports why it stopped.

## When you need it

A freshly built or packed structure is rarely at a force-field minimum: bond
lengths, angles, and close contacts carry excess energy. Before a production
simulation — or to compare energies meaningfully — you minimize the geometry so
the forces drop below a tolerance.

**`Lbfgs` moves atoms downhill on a set of potentials until the maximum force
falls under `fmax`.** The minimizer is the native limited-memory quasi-Newton
implementation, re-exported as `molpy.optimize.Lbfgs`; it
drives the `Potentials` that `PotentialCompiler` compiles from a force field for your frame.

## Minimizing a structure

```python
import molpy as mp

mol, _ = mp.conformer.Conformer(seed=42).generate(mp.io.smiles.read_smiles("CCO").to_atomistic())
typifier = mp.ff.typifier.OplsAaTypifier()
frame = typifier.typify(mol).to_frame()  # types, bonded terms, special pairs
forcefield = typifier.forcefield()  # OPLS-AA parameters of the types just assigned

potentials = mp.ff.PotentialCompiler(forcefield).compile(frame)  # bonded + pair terms
opt = mp.optimize.Lbfgs(potentials, fmax=0.05, max_steps=200)
report = opt.minimize(frame)  # relaxes frame's coordinates in place

print(report.converged, report.final_energy, report.final_fmax, report.n_steps)
```

`compile` prices the pair styles on every pair of atoms inside their cutoff
(OPLS-AA's 12 Å, which the typifier declares). The frame's special pairs —
its `pairs` block, which typing writes with the angles and dihedrals — are the
1-2, 1-3 and 1-4 pairs the force field's `special_bonds` weigh: OPLS-AA leaves
out 1-2 and 1-3 and halves 1-4. Every other pair is at full strength, so
overlapping atoms are pushed apart wherever they are on the molecule.

`minimize` relaxes the frame **in place** — afterwards `frame` holds the relaxed
coordinates — and returns an `OptimizationReport`. Copy the frame first if you
need the starting geometry too. A bool `atoms.free` column, when present, marks
the atoms that may move; fixed atoms stay put but still enter every energy and
force evaluation.

## Parameters

`Lbfgs(potentials, *, fmax=0.05, max_steps=500, max_step=0.2, memory=8)`:

| Parameter | Effect |
|---|---|
| `fmax` | Convergence threshold on the largest per-atom force magnitude (kcal/mol/Å). The run stops when every atom's force is below it. |
| `max_steps` | Hard cap on iterations — a safety net if `fmax` is never reached. |
| `max_step` | Largest displacement of any coordinate component per step (Å). Smaller = more stable but slower; raise it only if convergence is sluggish and stable. |
| `memory` | Number of past steps the L-BFGS Hessian approximation keeps. More memory = better curvature estimate, more storage. |

When you already hold coordinates outside a frame, `minimize_coords` relaxes a
writeable, C-contiguous float64 `(N, 3)` array in place and returns its report;
`minimize_batch` does the same for a `(B, N, 3)` stack, each structure
independently, and returns one report per structure.

## Reading the result

`OptimizationReport` carries `converged`, `final_energy`, `final_fmax`, `final_grad_rms` and `n_steps`.
Always check `converged`: a run with `converged = False` hit `max_steps` or
found no lower energy along its search direction, and has *not* reached the
minimum — loosen `fmax`, raise `max_steps`, or inspect the structure.

## Pitfalls

- **Not converged ≠ minimized.** A `False` `converged` means the run stopped
 at the step cap or its line search found no lower energy.
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
