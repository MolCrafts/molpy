# Python Quickstart

This quickstart follows a complete Python workflow: parse a molecule, generate
three-dimensional coordinates, convert to a `Frame`, attach a simulation box,
build a neighbor list, run RDF, evaluate MMFF94 energy, and write the result
to an XYZ file and a `*.mrec` record.

The goal is not to memorize every class. The goal is to see the boundary
between the graph representation (`Atomistic`) and the table representation
(`Frame`), because most molrs workflows cross that boundary deliberately.

## 1. Parse a Molecule

`molrs.io.smiles.SmilesIr` returns an intermediate representation. Convert it to
`Atomistic` when you want a graph with atoms and bonds.

```python
import molrs

ir = molrs.io.smiles.SmilesIr("CCO")  # ethanol
mol = ir.to_atomistic()

print("components:", ir.n_components)
print("heavy atoms:", mol.n_atoms)
print("bonds:", len(mol.bonds))
```

Expected output shape:

```text
components: 1
heavy atoms: 3
bonds: 2
```

At this stage there are no generated coordinates and implicit hydrogens are
not yet explicit graph nodes. That is why the next step runs the embedding
pipeline instead of trying to read `x`, `y`, and `z` columns from the graph.

## 2. Generate 3D Coordinates

Embedding converts topology into coordinates. Use a seed in examples so that
the result is reproducible across runs.

```python
mol3d, report = molrs.conformer.Conformer(speed="fast", seed=42).generate(mol)

print("atoms after embedding:", mol3d.n_atoms)
print("final energy:", report.final_energy)
print("stages:", [stage.stage for stage in report.stages])
```

`Conformer.generate` returns a plain `(mol, report)` tuple: construct the
generator once with your parameters, then call `generate` for each molecule.

## 3. Convert to a Frame

Writers and analyses operate on frames. A frame is a dictionary-like container
of named blocks. The `atoms` block holds coordinate columns and element data.

```python
frame = mol3d.to_frame()
atoms = frame["atoms"]

print("frame blocks:", frame.keys())
print("atom columns:", atoms.keys())
print("rows:", atoms.n_rows)
print("first x values:", atoms["x"][:3])
```

You should see an `atoms` block and usually a `bonds` block. Coordinate columns
are one-dimensional double-precision NumPy arrays. If you build a frame by
hand, use `np.float64` for scientific floating-point columns.

## 4. Attach a Periodic Box

Several analyses need to know whether coordinates are periodic. Attach a
`Box` before building neighbor lists if the system should be interpreted as a
periodic simulation cell.

```python
import numpy as np

frame.box = molrs.core.Box.cube(
    20.0,
    pbc=np.array([True, True, True], dtype=np.bool_),
)

print("box lengths:", frame.box.lengths)
print("box volume:", frame.box.volume())
```

The box is in the same length unit as your coordinates. molrs does not silently
convert between nanometer and angstrom conventions. Pick a unit convention for
the workflow and keep it consistent.

## 5. Build a Neighbor List and RDF

Neighbor search turns coordinates into pair lists. An analysis is configured
when it is built and given its data when it computes: `Rdf` takes its bins at
construction and one pair source (a `Neighbors` table or a built
`NeighborList`) per frame in `compute`, rather than doing its own distance
search or reading the frame.

```python
points = np.column_stack(
    [atoms["x"], atoms["y"], atoms["z"]]
).astype(np.float64, copy=False)

nl = molrs.core.NeighborList(6.0)
nl.build(points, frame.box)
neigh = nl.neighbors()

print("pairs:", neigh.n_pairs)
print("first pairs:", neigh.query_point_indices()[:5], neigh.point_indices()[:5])

from molrs.compute import Rdf
rdf = Rdf(64, 6.0, 0.0)                  # n_bins, r_max, r_min
rdf_result = rdf.compute(pairs=[neigh])  # one pair source per frame
print("rdf bins:", len(rdf_result.bin_centers))
print("first g(r):", rdf_result.rdf[:5])
```

For a single ethanol molecule in a large box, the RDF is just a small example
of the API shape. For real RDF work, build one neighbor list per trajectory
frame, with the same cutoff and boundary assumptions, and pass them all in one
`compute(pairs=[...])` call.

## 6. Evaluate MMFF94 Energy

Force-field evaluation starts from the molecular graph, not from arbitrary
coordinate tables. It takes three steps: a typifier labels the graph and
collects the parameters it assigned, the caller adds the non-bonded pair list,
and `PotentialCompiler` turns the force field and the typed frame into
potentials that can be evaluated.

```python
typifier = molrs.ff.typifier.Mmff94Typifier()
typed = typifier.typify(mol3d)
typed_frame = typed.to_frame()
print("typed blocks:", typed_frame.keys())

# forcefield() is a copy of exactly the types typify assigned.
ff = typifier.forcefield()
# Non-bonded terms need an explicit pairs block; the caller owns it.
typed_frame["pairs"] = molrs.ff.potential.intramolecular_pairs(typed_frame, ff)
potentials = molrs.ff.compile.PotentialCompiler(ff).compile(typed_frame)

energy, forces = potentials.calc_energy_forces(typed_frame)
print("energy:", energy)
print("coords shape:", typed_frame.coords.shape)
print("forces shape:", forces.shape)
```

Typing and compiling are separate steps on purpose. Typing gives a labeled
graph and accumulates the definitions it assigned in the typifier's output:
`forcefield()` returns a copy of it, `source_forcefield()` is the full parameter set it
matched against, and `typify` is its only writer.
`PotentialCompiler(ff).compile(frame)` is the one compile path every force
field uses, whether it came from a typifier or from a file.
`PotentialCompiler(ff).defer()` returns potentials that bind the frame they
are evaluated on, and `compile_typed(frame)` builds the neighbor-driven
kernels MD runs on. Compiling is stricter than typing: every term must resolve
to a supported parameter, so a molecule can type successfully while
compilation still reports incomplete coverage.

Coordinates and forces are both `(n_atoms, 3)`, so forces sum per atom
directly. For an isolated molecule they cancel:

```python
print("force balance:", np.abs(forces.sum(axis=0)).max())
```

A typifier of your own subclasses `molrs.ff.typifier.Typifier` and implements
only `assign(graph)`. It returns a `TypeAssignment` with one mapping of annotations per
atom and, in `links`, per term of any relation kind: keyed by a relation class
(`Bond`, `Angle`, …) or a kind name (`"bonds"`, or a custom
`"urey_bradleys"` the graph registered with `register_kind`, whose types land
in the category `urey_bradley`). A type annotation is
`(style, name, endpoints, params)`; endpoints are empty for an atom type. The
base class's `typify` copies the graph, stamps the assignment onto the copy and
defines the types in its output force field:

```python
from molrs.ff.typifier import TypeAssignment, Typifier


class EveryAtomX(Typifier):
    def assign(self, graph):
        return TypeAssignment(
            [{"type": ("full", "X", (), {"mass": 12.0})} for _ in graph.atoms],
            styles=[("atom", "full", {})],
        )


custom = EveryAtomX()
custom.typify(mol3d)
print([(s.category, s.name) for s in custom.forcefield().styles])
```

The built-in typifiers (`OplsAaTypifier`, `Mmff94Typifier`, …) can be
subclassed to carry your own attributes or methods, but they type in Rust:
a subclass that defines `assign` or `source_forcefield` raises `TypeError`. Start from
`Typifier` to supply your own typing.

## 7. Write Files

The I/O layer writes frames. This is the final boundary where the graph-based
work has become a portable coordinate table.

```python
molrs.io.write_xyz("ethanol.xyz", frame)
roundtrip = molrs.io.read_xyz("ethanol.xyz")
print("roundtrip atoms:", roundtrip["atoms"].n_rows)
```

The XYZ format stores coordinates and element symbols, but it does not preserve
the full force-field state or every topology detail. A
[record file](../guides/records.md) (`*.mrec`) keeps all of it — every block
and column at its dtype, typed metadata, the box, the force field, and whole
trajectories:

```python
molrs.io.write_mrec_frame("ethanol.mrec", typed_frame, forcefield=ff)
print(sorted(molrs.io.mrec.section_names("ethanol.mrec")))
```

## Summary

This quickstart crossed the main molrs boundaries:

- SMILES text became a graph-like `Atomistic`.
- `Conformer.generate` produced coordinates and diagnostics.
- `to_frame` produced the columnar representation used by I/O and analysis.
- `Box` supplied the boundary model for neighbor search.
- `Rdf` was configured with its bins and computed over explicit pair sources.
- `Mmff94Typifier` typed the graph, and `PotentialCompiler` compiled its
  force field into potentials for energy and force evaluation.
- `write_xyz` and `write_mrec_frame` wrote the result to disk.
