# Nanostructures

Nanotubes and graphene sheets are built by `mp.builder`, whose
`CarbonTubeBuilder` and `GrapheneBuilder` are molrs's. `CarbonTubeBuilder`
rolls an exact graphene lattice into a single-wall carbon nanotube;
`GrapheneBuilder` emits a flat honeycomb sheet. Each `build()` returns the
structure as a `Frame`; `mp.Atomistic.from_frame` makes it an editable graph.
The tube's shape is fixed when the builder is constructed;
unit-cell enumeration, seam closure, and connectivity are implementation
details.

Bulk crystals are molpy's `Lattice`, built from a CIF or from a cell and a
basis; see [Bulk crystals from a CIF](#bulk-crystals-from-a-cif).

## Zigzag, armchair, and chiral tubes

The usual `(n, m)` indices select the topology:

```python
import molpy as mp
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

Set `periodic=True` to close the axial bonds. `build()` returns the structure;
`cell()` returns the simulation cell those coordinates were laid out in, with
only the z axis periodic:

```python
builder = CarbonTubeBuilder(10, 10, length=50.0, periodic=True)
periodic = mp.Atomistic.from_frame(builder.build())
box = builder.cell(vacuum=12.0)

assert box.pbc.tolist() == [False, False, True]
assert all(len(periodic.incident_relations(atom.handle, "bonds")) == 3 for atom in periodic.atoms)
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

The builders create atoms and bonds only. Per-atom data can be written at
build time; angles and dihedrals are the graph's `generate_topology`, when you
want them:

```python
atoms_only = mp.Atomistic.from_frame(
    CarbonTubeBuilder(8, 0, cells=20).build(atom_type="CA", charge=0.0)
)
with_topology = mp.Atomistic.from_frame(CarbonTubeBuilder(8, 0, cells=2).build())
with_topology.generate_topology(gen_angle=True, gen_dihedral=True)

assert not list(atoms_only.angles)
assert list(with_topology.angles)
assert list(with_topology.dihedrals)
```

For a very large tube, keep the atoms-only graph through construction and let
the MD export workflow materialize higher-order topology when it is actually
needed.

## Bulk crystals from a CIF

A crystal is a `mp.builder.Lattice`: a `mp.Box` unit cell plus basis sites in
fractional coordinates. A CIF gives you the asymmetric unit and its space
group's operators; `mp.io.read_cif` keeps the operators in
`frame.meta["symops"]` and `Lattice.from_cif` expands every atom by them into
the whole conventional cell:

```python
# docs: skip — reads an offline CIF; Lattice.from_cif is unit-tested with fixtures
frame = mp.io.read_cif("NaCl.cif")  # Fm-3m: Na1 at 4a, Cl1 at 4b
salt = mp.builder.Lattice.from_cif(frame)
assert len(salt.basis) == 8  # 4 Na + 4 Cl per conventional cell

crystal = salt.build(repeats=(4, 4, 4))
frame = crystal.to_frame()
frame.box = salt.supercell((4, 4, 4))
```

The cell is the file's own box, so a triclinic cell keeps its orientation:
`box.h` holds the lattice vectors as columns. `from_cif` raises `ValueError`
rather than guess when the frame has no box, when the file has no operator
loop (it does not assume P1; list `x,y,z` for a P1 structure), or when a site
is partially occupied, since a lattice places one atom per site.

To cut a cluster or slab instead of a block of whole cells, pass a region;
the tile covers its whole bounding box, on every side of the origin:

```python
nickel = mp.builder.Lattice.fcc(a=3.52, species="Ni")
cluster = nickel.build(mp.core.Sphere([0.0, 0.0, 0.0], 5.0))
assert len(cluster.atoms) == 55  # the centre atom and its first four shells
```
