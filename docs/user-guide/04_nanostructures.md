# Nanostructures

Nanotubes and graphene sheets are built in **molrs** (`molrs.builder`) and
re-exported by MolPy. `CarbonTubeBuilder` rolls an exact graphene lattice into a
single-wall carbon nanotube; `GrapheneBuilder` emits a flat honeycomb sheet.
The tube's shape is fixed when the builder is constructed;
unit-cell enumeration, seam closure, and connectivity are implementation
details.

## Zigzag, armchair, and chiral tubes

The usual `(n, m)` indices select the topology:

```python
from molpy.builder import CarbonTubeBuilder, GrapheneBuilder

zigzag = CarbonTubeBuilder(8, 0, length=30.0).build()
armchair = CarbonTubeBuilder(6, 6, cells=4).build()
chiral = CarbonTubeBuilder(6, 3, cells=3).build()

sheet = GrapheneBuilder(8, 8, periodic_xy=True).build()
```

`length` rounds up to complete translational cells. Use `cells` when the exact
number of cells matters; the two arguments are mutually exclusive. All three
tubes are open along the axis and have dangling end valences.

## Periodic tubes

Set `periodic=True` to close the axial bonds. `build()` returns the molecular
graph; `cell()` returns the simulation cell those coordinates were laid out in,
with only the z axis periodic:

```python
builder = CarbonTubeBuilder(10, 10, length=50.0, periodic=True)
periodic = builder.build()
box = builder.cell(vacuum=12.0)

assert box.pbc.tolist() == [False, False, True]
assert all(len(periodic.get_neighbors(atom)) == 3 for atom in periodic.atoms)
```

The two are separate products because a molecular graph is topology and
chemistry — the cell describes the simulation, and its one home is `frame.box`:

```python
frame = periodic.to_frame()
frame.box = box
```

The circumference is part of the molecular topology, not a simulation-box
periodic direction. Bonds come from the rolled graphene lattice rather than a
Cartesian distance cutoff, so the seam is exact for zigzag, armchair, and
general chiral tubes.

## Atom annotations and deferred topology

The scalable default creates atoms and bonds only. Per-atom data can be written
at build time, while angles and dihedrals remain optional:

```python
atoms_only = CarbonTubeBuilder(8, 0, cells=20).build(atom_type="CA", charge=0.0)
with_topology = CarbonTubeBuilder(8, 0, cells=2).build(finalize="topology")

assert not list(atoms_only.angles)
assert list(with_topology.angles)
assert list(with_topology.dihedrals)
```

For a very large tube, keep the atoms-only graph through construction and let
the MD export workflow materialize higher-order topology when it is actually
needed.
