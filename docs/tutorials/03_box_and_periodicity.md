# Box and Periodicity

Two atoms sit at $x = 1$ and $x = 9$ in a cell of length 10. Are they 8 Å
apart, or 2 Å through the periodic image?

Without the cell, the question is undefined. **`Box` is the simulation cell**:
it wraps positions into the primary image and computes minimum-image distances.

What it is **not**: a dump format, a neighbour list, or a substitute for
unwrapped trajectories in displacement analysis (see
[MSD](../compute/msd.md)).

## Why periodicity matters

Bulk MD uses a small sample and tiles space with *periodic boundary conditions*
so the sample has no free surface. An atom leaving one face re-enters the
opposite face. The distance that enters physics is always the shortest path —
which may cross a boundary.

Raw coordinates alone are therefore ambiguous. Every distance, neighbour list,
and structural analysis must know the box. MolPy keeps that object explicit:
periodicity is part of the physical model, not a hidden flag.

## Creating a box

Factories cover the common cells: cubic and orthorhombic. A triclinic cell is
built from its matrix, which `Box.matrix_from_lengths_tilts` assembles from edge
lengths and LAMMPS tilt factors.

```python
import molpy as mp
import numpy as np

cubic = mp.Box.cube(20.0)
ortho = mp.Box.ortho(np.array([10.0, 20.0, 30.0]))
tric = mp.Box(
    mp.Box.matrix_from_lengths_tilts(
        np.array([10.0, 12.0, 15.0]), np.array([1.0, 0.5, 0.2])
    )
)

print(cubic.style, ortho.style, tric.style) # orthogonal orthogonal triclinic
```

You can also pass a 3×3 matrix directly. Columns are lattice vectors.

```python
matrix = np.array([[10.0, 1.0, 0.5],
 [0.0, 12.0, 0.2],
 [0.0, 0.0, 15.0]])
box = mp.Box(matrix=matrix)
print(box.lengths)
```

Every box carries a `pbc` array — three booleans controlling which axes are periodic. The default is fully periodic. For a slab geometry, turn off the z axis.

```python
slab = mp.Box.ortho(np.array([20.0, 20.0, 50.0]), pbc=np.array([True, True, False]))
print(slab.pbc) # [ True True False]
```

## Derived properties

A box exposes geometric quantities computed from the cell matrix `h` (lattice vectors as columns): `lengths`, `volume()`, `origin`, `bounds`, and for triclinic cells, `tilts` and `angles`.

```python
box = mp.Box.ortho(np.array([10.0, 12.0, 15.0]))
print(f"lengths: {box.lengths}")
print(f"volume: {box.volume()}")
print(f"style: {box.style}")
```

## Wrapping coordinates into the primary cell

Atoms that have drifted outside the box during a simulation can be mapped back with `wrap`. This produces wrapped positions in the primary image.

```python
box = mp.Box.cube(10.0)

points = np.array([
 [12.0, -2.0, 5.0],
 [25.0, 8.0, -3.0],
])

wrapped = box.wrap(points)
print(wrapped)
# Points are now inside [0, 10) on each axis
```

If you need to reconstruct the unwrapped trajectory later, `images` tells you how many box lengths each coordinate was shifted, and `unwrap` reverses the operation.

```python
images = box.images(points)
unwrapped = box.unwrap(wrapped, images)
print(np.allclose(unwrapped, points)) # True
```

## Fractional coordinates

Converting between absolute (Cartesian) and fractional coordinates is sometimes useful for analysis or for writing certain file formats. Fractional coordinates express positions as fractions of the lattice vectors, so they always lie in [0, 1) for wrapped systems.

```python
absolute = np.array([[5.0, 3.0, 7.0]])
fractional = box.to_frac(absolute)
restored = box.to_cart(fractional)

print(fractional) # [[0.5, 0.3, 0.7]]
print(np.allclose(restored, absolute)) # True
```

## Minimum-image distances

In a periodic system, the physically meaningful separation between two points is the shortest one — the minimum-image displacement. `delta(..., minimum_image=True)` computes the displacement vector from the first point set to the second; `distances` computes the scalar distance.

```python
box = mp.Box.cube(10.0)

r1 = np.array([[1.0, 1.0, 1.0]])
r2 = np.array([[9.5, 9.5, 9.5]])
```

Without periodic awareness, these two points appear to be about 14.7 Å apart. Under minimum-image convention, the shortest path crosses the periodic boundary and the real distance is much smaller.

```python
dr = box.delta(r1, r2, minimum_image=True)
d = box.distances(r1, r2)

print(f"displacement: {dr}")
print(f"distance: {d}")
```

For pairwise distances between two sets of points, `pairwise_distances` returns an (N, M) matrix.

```python
set_a = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
set_b = np.array([[9.5, 9.5, 9.5], [8.0, 8.0, 8.0]])
distances = box.pairwise_distances(set_a, set_b)
print(distances.shape) # (2, 2)
print(distances)
```

## Box on Frame

A box is attached to a Frame as `frame.box`, not stored in metadata. This is the standard way to associate a simulation cell with molecular data.

```python
# docs: skip — reads system.data offline artifact; I/O unit-tested with fixtures
frame = mp.Frame(blocks={
 "atoms": {"x": [1.0, 9.5], "y": [1.0, 9.5], "z": [1.0, 9.5]},
})
frame.box = mp.Box.cube(10.0)

# I/O readers set frame.box automatically
frame = mp.io.read_lammps_data("system.data", atom_style="full")
print(frame.box.lengths) # from the data file header
```

All compute operators (MSD, RDF, etc.) read the box from `frame.box`.

## When the box matters

Use `Box` as soon as your system is meant to be periodic. Do not wait for engine export to start thinking about it. The box determines how coordinates are interpreted — wrap, delta, and distances all depend on it. Any analysis on a periodic system that ignores the box is silently wrong.

Once a single snapshot is not enough and your workflow tracks the system through time, the next abstraction is a trajectory.

See also: [Block and Frame](02_block_and_frame.md), [Trajectory](05_trajectory.md).
