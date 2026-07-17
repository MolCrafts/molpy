# Nanostructures

`CarbonTubeBuilder` constructs a single-wall carbon nanotube directly from its
graphene topology. The user-facing operation is one `build()` call; chirality,
unit-cell enumeration, seam closure, and connectivity are implementation
details.

## Zigzag, armchair, and chiral tubes

The usual `(n, m)` indices select the topology:

```python
from molpy.builder import CarbonTubeBuilder

builder = CarbonTubeBuilder()

zigzag = builder.build(8, 0, length=30.0)
armchair = builder.build(6, 6, cells=4)
chiral = builder.build(6, 3, cells=3)
```

`length` rounds up to complete translational cells. Use `cells` when the exact
number of cells matters; the two arguments are mutually exclusive. All three
tubes are open along the axis and have dangling end valences.

## Periodic tubes

Set `periodic=True` to close the axial bonds and mark only the box's z axis as
periodic:

```python
periodic = builder.build(10, 10, length=50.0, periodic=True, vacuum=12.0)

assert periodic["box"].pbc.tolist() == [False, False, True]
assert all(len(periodic.get_neighbors(atom)) == 3 for atom in periodic.atoms)
```

The circumference is part of the molecular topology, not a simulation-box
periodic direction. Bonds come from the rolled graphene lattice rather than a
Cartesian distance cutoff, so the seam is exact for zigzag, armchair, and
general chiral tubes.

## Atom annotations and deferred topology

The scalable default creates atoms and bonds only. Per-atom data can be written
at build time, while angles and dihedrals remain optional:

```python
atoms_only = builder.build(8, 0, cells=20, atom_type="CA", charge=0.0)
with_topology = builder.build(8, 0, cells=2, finalize="topology")

assert not list(atoms_only.angles)
assert list(with_topology.angles)
assert list(with_topology.dihedrals)
```

For a very large tube, keep the atoms-only graph through construction and let
the MD export workflow materialize higher-order topology when it is actually
needed.
