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
data: {$file: series/voronoi/argon_volumes.json}
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
standard deviation 4.5 Å³ — about 9 %.
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
`VoronoiDomainAnalysis` builds on.

## Computing it

A tessellation of one snapshot needs three pieces of data — the positions, the
box, and one radius per atom — and no configuration at all, so `RadicalVoronoi`
is built empty and everything goes into `compute`:

```python
import numpy as np
import molpy as mp
from molpy.compute import RadicalVoronoi

rng = np.random.default_rng(0)
n_atoms, box_length = 400, 20.0
xyz = rng.uniform(0.0, box_length, size=(n_atoms, 3))
box = mp.Box.cube(box_length)
radii = np.zeros(n_atoms)                       # equal radii -> plain Voronoi

cells = RadicalVoronoi().compute(positions=xyz, simbox=box, radii=radii)
volumes = cells.volumes

print(volumes.shape)                                    # -> (400,)
print(round(float(volumes.sum()), 1), box_length**3)    # -> 8000.0 8000.0
print(round(float(volumes.mean()), 2))                  # -> 20.0
```

The volumes sum to the box volume to the digit, and the mean is $V/N = 20$ Å³.
Run that assertion on your own system before anything else: if it fails, the box
or the coordinates are wrong and nothing downstream is worth reading. On a
trajectory, read `frame.coords` and `frame.box` from each frame and tessellate
it; each snapshot is one call.

`cells.neighbors(i)` lists the cells face-adjacent to cell `i`, `cells.faces[i]`
the faces themselves with their areas, and `cells.total_volume()` the sum. For
a mixture, pass real radii — that is the point of the radical construction:

```python
mixed = rng.uniform(0.5, 1.5, n_atoms)
cells_mixed = RadicalVoronoi().compute(positions=xyz, simbox=box, radii=mixed)
print(float(np.corrcoef(mixed, cells_mixed.volumes)[0, 1]) > 0.3)  # -> True
```

Cell volume now correlates with particle radius. With `radii = 0` it does not.

### Domains and voids are computes on the cells

The tessellation is a result that other computes consume. Face adjacency is a
neighbour definition with no cutoff, so the natural next step is to merge
face-adjacent cells into regions.

`VoronoiDomainAnalysis` does it for a per-atom classification. It takes the
cells and a **partition** — one unsigned label per atom, from your own
`mol_id`-style column, a `SolidLiquid` classification from [Order](order.md),
or a [`Cluster`](cluster.md) or `Kmeans` result — and merges face-adjacent
cells that share a label into connected domains, without inventing another
cutoff. Split the random box into a left and a right half:

```python
from molpy.compute import VoronoiDomainAnalysis

side = (xyz[:, 0] < box_length / 2).astype(np.uint32)
domains = VoronoiDomainAnalysis().compute(cells=cells, partition=side)
print(domains.count, sorted(domains.sizes.tolist()))    # -> 2 [187, 213]
```

Two labels, two connected slabs. `domain_of` gives each cell's domain and
`largest_fraction` the share of cells in the biggest one; the result is itself
a partition, so the domains can be measured next with the group computes on
[Shape](shape.md).

`VoronoiVoidAnalysis` does the same for empty space: give it a per-cell boolean
mask marking which cells are void — probe particles, or simply the loosest
cells — and the box volume, and it merges the marked cells into cavities:

```python
from molpy.compute import VoronoiVoidAnalysis

loose = volumes > np.quantile(volumes, 0.9)
voids = VoronoiVoidAnalysis().compute(
    cells=cells, is_void=loose.tolist(), box_volume=box_length**3
)
print(np.round(voids.cavity_volumes, 1).tolist(), round(voids.void_fraction, 3))
# -> [1491.4, 33.4] 0.191
```

The 40 loosest cells of a random point set are not scattered: almost all of
them touch, forming one large cavity. That is the parameter-free route to
porosity in a framework material.

### Integrating an electron density

`VoronoiIntegration` partitions a volumetric electron density over the cells and
returns per-molecule charges and dipole moments. It is the standard route from
*ab initio* MD to infrared intensities, because the dipole flux the
[Spectra](spectra.md) page needs has to come from somewhere.

Its only configuration is the length unit of the density grid (`"angstrom"`
or `"bohr"` — a cube file is usually in bohr, and the unit is never guessed).
Its data is everything that describes the snapshot: positions, radii and
nuclear charges of the atoms, the partition that says which atom belongs to
which molecule, the box, the `RegularGrid` the density lives on, and the
density values themselves. A uniform density that holds one electron per atom
must leave every molecule neutral:

```python
from molpy.compute import RegularGrid, VoronoiIntegration

cell = mp.Box.cube(4.0)
sites = np.array([[x, y, z] for x in (1.1, 3.1) for y in (1.1, 3.1) for z in (1.1, 3.1)])
molecule = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint32)
grid = RegularGrid.over_box(cell, [16, 16, 16])
density = np.full((16, 16, 16), 8 / cell.volume())     # one electron per atom

moments = VoronoiIntegration("angstrom").compute(
    positions=sites, radii=np.ones(8), nuclear_charges=np.ones(8),
    partition=molecule, simbox=cell, grid=grid, density=density,
)
print(moments.groups.tolist(), np.round(moments.charges, 6).tolist())  # -> [0, 1] [0.0, 0.0]
```

`moments.dipoles` holds one dipole per molecule, and two such results — the
density under a field $+E$ and under $-E$ — are what `FiniteFieldPolarizability`
differentiates into a polarizability.

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
