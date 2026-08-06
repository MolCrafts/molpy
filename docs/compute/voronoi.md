# Voronoi tessellation

[`LocalDensity`](density.md) asks how many atoms are inside a sphere you chose.
That "you chose" is the weakness: the answer depends on a radius nobody can
derive from the physics.

A Voronoi tessellation removes the choice. Assign every point in space to its
**nearest** atom, and the box is partitioned into one convex cell per atom, with
no gaps and no overlaps. Each atom now owns a volume, a set of faces, and a list
of face-sharing neighbours — all without a cutoff.

The **radical** (Laguerre) variant generalizes this to particles of different
size: the boundary between two atoms shifts according to their radii, so a big
atom gets a bigger cell. With equal radii it reduces to the ordinary Voronoi
tessellation.

## Volume is exactly conserved

Because the cells tile space, their volumes must sum to the box volume. That is
not an approximation, and it makes the mean cell volume something you know
before computing it:

$$
\langle v \rangle = \frac{V}{N} = \frac{1}{\rho}.
$$

For the argon box — 500 atoms in 24 139 Å³ — that is 48.28 Å³ per atom, and the
tessellation returns exactly that. The *interesting* number is the width of the
distribution.

<figure id="fig-voronoi-vol" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:10"
data: {$file: data/voronoi/argon_volumes.json}
mark: {type: line, strokeWidth: 2.4, interpolate: monotone}
encoding:
  x:
    field: v
    type: quantitative
    title: "volume (Å³)"
  y:
    field: p
    type: quantitative
    title: "P(v)"
```

</div>

**Figure 1.** Radical-Voronoi cell volumes in liquid argon at 85 K (equal
radii, so an ordinary Voronoi tessellation). Mean 48.28 Å³ = $V/N$ exactly,
standard deviation 4.4 Å³ — about 9 %.
</figure>

A 9 % spread around a mean fixed by construction is the free-volume distribution
of the liquid: how unevenly the available space is shared out. Cells in the
upper tail are the loosely packed sites where a diffusive hop is most likely to
begin, which makes this a structural precursor to the dynamics on the
[MSD](msd.md) and [Persist](persist.md) pages. A crystal would give a single
spike; a broad tail is a signature of disorder.

Two further things come free. The **number of faces** of a cell is a
parameter-free coordination number — no cutoff, no first-minimum argument. And
**face adjacency** is a natural definition of "neighbour", which is what
`voronoi_domains` builds on.

## Computing it

`RadicalVoronoi` takes `(positions, radii, box)`. Note that despite what its
docstring says, you call the object itself; there is no `.compute` method.

```python
import numpy as np
import molpy as mp
from molpy.compute import RadicalVoronoi

rng = np.random.default_rng(0)
n_atoms, box_length = 400, 20.0
xyz = np.ascontiguousarray(rng.uniform(0.0, box_length, size=(n_atoms, 3)))
radii = np.zeros(n_atoms)                       # equal radii -> plain Voronoi

cells = RadicalVoronoi()(xyz, radii, mp.Box.cubic(box_length))
volumes = np.asarray(cells.volumes)

print(volumes.shape)                                    # -> (400,)
print(round(float(volumes.sum()), 1), box_length**3)    # -> 8000.0 8000.0
print(round(float(volumes.mean()), 2))                  # -> 20.0
```

The volumes sum to the box volume to the digit, and the mean is $V/N = 20$ Å³.
Run that assertion on your own system before anything else: if it fails, the box
or the coordinates are wrong and nothing downstream is worth reading.

`cells.neighbors` holds the face-adjacency lists and `cells.total_volume` the
sum. For a mixture, pass real radii — that is the point of the radical
construction:

```python
mixed = rng.uniform(0.5, 1.5, n_atoms)
big = np.asarray(RadicalVoronoi()(xyz, mixed, mp.Box.cubic(box_length)).volumes)
print(float(np.corrcoef(mixed, big)[0, 1]) > 0.3)       # -> True
```

Cell volume now correlates with particle radius. With `radii = 0` it does not.

### Domains and voids

`voronoi_domains` merges face-adjacent cells that share a label — use it to turn
a per-atom classification (say solid/liquid from [Order](order.md)) into
connected regions without inventing another cutoff. It returns `sizes`,
`count`, `largest_fraction`, and `domain_of`.

`voronoi_voids` does the same for empty space: give it a per-cell boolean mask
marking probe cells as void, and it merges them into cavities, returning
`cavity_volumes`, `total_void_volume`, and `void_fraction`. That is the
parameter-free route to porosity in a framework material.

### Integrating an electron density

`VoronoiIntegration` partitions a volumetric electron density over the cells and
returns per-molecule charges and dipole moments. It is the standard route from
*ab initio* MD to infrared intensities, because the dipole flux the
[Spectra](spectra.md) page needs has to come from somewhere. Call it with
`(positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box)`.

Its accuracy is set by the density grid spacing, and the convergence test is to
halve the spacing until the total charge and the resulting spectrum stop moving.
Report the spacing alongside any published dipole or IR intensity.

## When it goes wrong

**The volumes do not sum to the box volume.**
The box does not match the coordinates, or the coordinates are not wrapped into
it. This check is exact; treat a failure as fatal rather than as noise.

**A few cells are enormous.**
A missing or duplicated atom leaves a large region for its neighbours to claim.
Look for coincident coordinates.

**The distribution is much broader than about 10 %.**
Either the system is genuinely heterogeneous — an interface, a void, a mixture —
or it is dilute, where a cell's volume is dominated by the happenstance position
of one neighbour.

**Radical and ordinary tessellations differ a lot.**
Expected if your radii really differ. If they should not, check you did not pass
diameters where radii were wanted.

**Face counts do not match the coordination number from [$g(r)$](rdf.md).**
They measure different things. Voronoi counts every face, including slivers
contributed by distant atoms; the $g(r)$ coordination number counts atoms inside
a radius. Weighting faces by area brings the two much closer.

## Check yourself

- Sum the volumes and compare with the box volume. Exact agreement, every time.
- Confirm the mean equals $V/N$, then look only at the width — the mean carries
  no information about your system.
- Tessellate a perfect FCC lattice: every cell should have identical volume, and
  12 faces.

## References

- G. Voronoi, *J. Reine Angew. Math.* **134**, 198 (1908) — the tessellation.
- B. J. Gellatly, J. L. Finney, *J. Non-Cryst. Solids* **50**, 313 (1982) — the
  radical construction for unequal spheres.
- M. Thomas, M. Brehm, B. Kirchner, *Phys. Chem. Chem. Phys.* **17**, 3207
  (2015) — Voronoi partitioning of electron density for IR spectra.

## See also

- [Density](density.md) — the cutoff-dependent alternative
- [Cluster](cluster.md) — connectivity from a cutoff rather than from faces
- [Spectra](spectra.md) — what `VoronoiIntegration` feeds
- [API reference](../api/compute.md)
