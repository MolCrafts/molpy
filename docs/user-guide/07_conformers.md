# 3D Conformer Generation

Between SMILES and simulation stands one hard step: real coordinates.
`Conformer` takes it with the conformer generator — and reports every stage of the
attempt.

## What conformer generation solves

A parsed molecule is a *graph*: elements and bonds, but no geometry. Almost
everything downstream — typification's SMARTS matching that needs 3D, packing,
force-field evaluation, export to a simulation engine — needs real coordinates.

**`Conformer` embeds a graph into 3D with ETKDGv3 — RDKit's `EmbedMolecule` —
and returns a fresh `Atomistic` plus a per-stage report.** Hydrogens are added
as needed and the input graph is never mutated. Like `EmbedMolecule` it runs no
force field: relaxing the result is a separate step (below).

## Generating a conformer

```python
import molpy as mp

mol = mp.io.smiles.read_smiles("CCO").to_atomistic() # ethanol graph (heavy atoms only)
mol_3d, report = mp.conformer.Conformer(seed=42).generate(mol)

print(mol_3d.n_atoms) # 9 — heavy atoms + added hydrogens
```

`generate` returns a **tuple**: the new `Atomistic` (with coordinates and any
added hydrogens) and a `ConformerReport`. The input `mol` is left untouched, so
you can generate several independent conformers from the same graph.

## Constructor parameters

`mp.conformer.Conformer` takes RDKit's `EmbedParameters` names; an empty
molecule is refused with `ValueError`.

| Parameter | Default | Meaning |
|---|---|---|
| `add_hydrogens` | `True` | Fill valences with explicit hydrogens before embedding (the same edit as `mp.perceive.add_hydrogens`). Leave on unless your graph already carries all H. |
| `max_iterations` | `0` | Embedding attempts before giving up; `0` is RDKit's `10 × n_atoms`. |
| `seed` | `None` | RNG seed. **Set it for reproducible geometries** — omitting it gives a different conformer each run. |
| `use_random_coords` | `False` | Start each attempt from random coordinates instead of the eigenvector embedding of a random distance matrix. |
| `use_exp_torsion_angle_prefs` | `True` | The ETKDG experimental-torsion terms ("ET"). |
| `use_basic_knowledge` | `True` | Flat aromatic rings and planar sp2 centres ("K"). With ET and K both off the embedding is plain distance geometry. |

A dummy atom (`*`, atomic number 0 — a polymer repeat unit's attachment point)
is embedded as RDKit embeds one: its radii are 0 and it has no UFF type, so its
bond takes the covalent-radius estimate (C–`*` 0.76 Å).

## Long chains

Each attempt is judged before it is kept. The first distance-geometry
minimization must end below RDKit's energy gate, and the 3D refinement must
not leave a bond angle above 170° at a centre that is not sp: on a long
flexible chain the refinement can relieve its torsion strain by straightening
an angle — a C–O–C of a PEO 25-mer at 180°, where the torsions through it are
undefined — and such an attempt is discarded for the next one. The
eigenvector starts of a chain that long fail the first gate on every attempt,
as they do in RDKit, whose advice for large molecules is
`use_random_coords=True`; the embedding then runs the same number of attempts
from random coordinates and records `"used random-coordinate fallback
embedding"` in `report.warnings`.

## Relaxing with MMFF94

The embedding is the distance-geometry structure. For a force-field minimum,
compose the relaxation as RDKit's `MMFFOptimizeMolecule` follows `EmbedMolecule`:

```python
typifier = mp.ff.typifier.Mmff94Typifier()
frame = typifier.typify(mol_3d).to_frame()
potentials = mp.ff.PotentialCompiler(typifier.forcefield()).compile(frame)
mp.optimize.Lbfgs(potentials, fmax=0.05, max_steps=1000).minimize(frame)  # in place
```

Charged atoms must already carry the canonical integer `"formal_charge"` key
(the parsers emit it) so the generator fills the right hydrogen count for `[N+]` / `[N-]`.

## Reading the report

`ConformerReport` aggregates the run:

- `stages` — a list of `ConformerStageReport`, one per pipeline stage.
- `warnings` — any non-fatal issues raised during generation.

Each `ConformerStageReport` records `stage`, `steps`, `converged`,
`energy_before` and `energy_after`:

```python
for s in report.stages:
 status = "converged" if s.converged else "hit step limit"
 # A stage that never ran an energy model reports None rather than 0.0.
 before = "n/a" if s.energy_before is None else f"{s.energy_before:.3f}"
 after = "n/a" if s.energy_after is None else f"{s.energy_after:.3f}"
 print(f"{s.stage}: {before} -> {after} ({s.steps} steps, {status})")
```

A stage that reports `converged = False` means it exhausted its step budget —
the geometry is usable but not fully relaxed.

## 2D coordinates for a sketch

A drawing needs different coordinates from a simulation: a flat layout where
rings are regular polygons and chains zigzag, not the puckered 3D geometry of
the embedder. `mp.depict.Depictor` computes it from the graph alone and keeps
every property, so formal charges and ports reach a 2D viewer with the
positions:

```python
cation = mp.io.smiles.read_smiles_fragment("[<]c1ccc(CC[N+](C)(C)C)cc1").to_template()
flat = mp.depict.Depictor().layout(cation)
sketch = flat.to_frame()
print(sketch["atoms"]["z"].max(), sketch["atoms"]["formal_charge"].max())
```

The layout is deterministic, uses one bond length (1.5 by default) and
replaces any coordinates the input had; keep the 3D molecule if you need
both.

## Pitfalls

- **No `seed` = non-reproducible.** Two runs give different (both valid)
 conformers. Pin `seed` whenever you compare or cache geometries.
- **A graph with no atoms raises `ValueError`.** Parse before you generate.
- This is a *single*-conformer embedder, not a conformer-ensemble search; call
 it repeatedly with different seeds if you need diversity.

## See also

- [Parsing Chemistry](01_parsing_chemistry.md) — building the input graph.
- [Force Field Typification](06_typifier.md) — the next step, which needs 3D.
- [Geometry Optimization](08_geometry_optimization.md) — force-field
 relaxation once the molecule has a force field.
