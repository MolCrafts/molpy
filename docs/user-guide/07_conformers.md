# 3D Conformer Generation

This guide explains how MolPy turns a 2D molecular graph (from SMILES or any
`Atomistic`) into a physically reasonable set of 3D coordinates, and how to read
the report that comes back.

## What conformer generation solves

A parsed molecule is a *graph*: elements and bonds, but no geometry. Almost
everything downstream — typification's SMARTS matching that needs 3D, packing,
force-field evaluation, export to a simulation engine — needs real coordinates.

**`Conformer` embeds a graph into 3D using the molrs Rust generator and returns a
fresh `Atomistic` plus a per-stage report.** Hydrogens are added as needed and
the input graph is never mutated (molrs clones it internally).

## Generating a conformer

```python
from molpy.parser import parse_molecule
from molpy.conformer import Conformer

mol = parse_molecule("CCO")                    # ethanol graph (heavy atoms only)
mol_3d, report = Conformer(seed=42).generate(mol)

print(mol_3d.n_atoms)          # 9  — heavy atoms + added hydrogens
print(report.final_energy)     # energy of the returned structure
```

`generate` returns a **tuple**: the new `Atomistic` (with coordinates and any
added hydrogens) and a `ConformerReport`. The input `mol` is left untouched, so
you can generate several independent conformers from the same graph.

## Constructor parameters

`Conformer` subclasses `molrs.Conformer`; the constructor parameters are
inherited unchanged:

| Parameter | Meaning |
|---|---|
| `speed` | Speed/quality trade-off for the embedding + refinement passes. Faster settings do fewer refinement steps. |
| `add_hydrogens` | Whether to fill valences with explicit hydrogens before embedding. Leave on unless your graph already carries all H. |
| `seed` | RNG seed for the stochastic embedding. **Set it for reproducible geometries** — omitting it gives a different conformer each run. |

Charged atoms must already carry the canonical integer `"formal_charge"` key
(the parsers emit it) so molrs fills the right hydrogen count for `[N+]` / `[N-]`.

## Reading the report

`ConformerReport` aggregates the run:

- `final_energy` — energy of the returned structure.
- `stages` — a list of `ConformerStageReport`, one per pipeline stage.
- `warnings` — any non-fatal issues raised during generation.

Each `ConformerStageReport` records `stage`, `steps`, `converged`,
`energy_before`, `energy_after`, and `elapsed_ms`:

```python
for s in report.stages:
    status = "converged" if s.converged else "hit step limit"
    print(f"{s.stage}: {s.energy_before:.3f} -> {s.energy_after:.3f} "
          f"({s.steps} steps, {status})")
```

A stage that reports `converged = False` means it exhausted its step budget —
the geometry is usable but not fully relaxed; try a slower `speed`.

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
