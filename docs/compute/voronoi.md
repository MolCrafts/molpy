# Voronoi

This page is a self-contained, textbook-style introduction to the **radical
Voronoi tessellation** and the three analyses built on it: connected-domain
detection, void analysis, and electron-density integration into per-molecule
charges and dipoles. A Voronoi tessellation assigns every point in space to a
site (atom); the **radical** (Laguerre / power) variant weights the partition by
atomic radii — the physically correct division for differently sized atoms.

The tessellation and reductions run in the high-performance backend; they operate
directly on positions, radii, and the box rather than on `Frame` objects.

!!! note "Conventions used throughout"
    - Positions and radii in Å; volumes in Å³; charges in $e$; dipoles in
      $e\cdot$Å.
    - Fully periodic (minimum-image) tessellation.
    - A *cell* is one atom's radical-Voronoi polyhedron; *domains* and *voids*
      are groupings of cells.

---

## 1. Ordinary Voronoi vs radical (power) Voronoi

### 1.1 Ordinary Voronoi

For sites $\{\mathbf{r}_i\}$, the ordinary Voronoi cell of $i$ is

$$
V_i = \bigl\{\mathbf{x}\ \big|\ |\mathbf{x}-\mathbf{r}_i| \le |\mathbf{x}-\mathbf{r}_j|\ \forall j\bigr\}.
$$

The boundary between $i$ and $j$ is the perpendicular bisector of
$\mathbf{r}_i\mathbf{r}_j$. That is correct only when all atoms have the **same
size**. In a polydisperse liquid, large atoms are systematically
**under-assigned** volume and small atoms over-assigned.

### 1.2 Power distance and radical planes

The **power** (Laguerre) distance from a point $\mathbf{x}$ to a ball of center
$\mathbf{r}_i$ and radius $R_i$ is

$$
\boxed{\;
\pi_i(\mathbf{x}) = |\mathbf{x}-\mathbf{r}_i|^2 - R_i^2
\;}
$$

The radical cell of $i$ is

$$
V_i^\mathrm{rad}
= \bigl\{\mathbf{x}\ \big|\ \pi_i(\mathbf{x})\le \pi_j(\mathbf{x})\ \forall j\bigr\}.
$$

Equating $\pi_i=\pi_j$ gives the **radical plane** between $i$ and $j$:

$$
2\mathbf{x}\cdot(\mathbf{r}_j-\mathbf{r}_i)
= |\mathbf{r}_j|^2 - |\mathbf{r}_i|^2 - R_j^2 + R_i^2.
$$

The plane is still perpendicular to $\mathbf{r}_j-\mathbf{r}_i$, but it is
**shifted** toward the smaller sphere by an amount fixed by $R_i^2-R_j^2$. When
$R_i=R_j$ the construction collapses to ordinary Voronoi.

### 1.3 Why this is the right partition for chemistry

- Cell volume $v_i = |V_i^\mathrm{rad}|$ is a meaningful **local volume** for
  packing fraction $\eta_i = (4\pi/3)R_i^3 / v_i$ and local density $1/v_i$.
- Face-sharing graph of the tessellation is a natural neighbour list for
  polydisperse systems (no single hard cutoff).
- Integrating a density field over $V_i^\mathrm{rad}$ partitions continuous
  electron density into atomic/molecular contributions without arbitrary
  spherical cutoffs.

The examples below share this setup:

```python
import numpy as np
import molpy as mp

rng = np.random.default_rng(0)
positions = rng.uniform(0.0, 20.0, size=(200, 3))
radii = np.full(len(positions), 1.0)
box = mp.Box.cubic(20.0)
```

```python
from molpy.compute import RadicalVoronoi

cells = RadicalVoronoi()(positions, radii, box)  # -> VoronoiCells
cells.neighbors(0)  # cells sharing a face with cell 0
```

### 1.4 Radii choice

The partition depends on $\{R_i\}$. Common consistent sets:

| Set | Typical use |
|---|---|
| van der Waals | condensed liquids, packing |
| covalent / crystal | bonded networks |
| ionic (Shannon) | molten salts, ionic liquids |

Report the radii table with every result. Mixing inconsistent tables between
systems invalidates volume comparisons.

---

## 2. Domains group cells that belong together

Labelling each atom (by species, charge sign, polarity, head/tail of a
surfactant, …) and merging face-sharing cells with the **same** label yields
**domains** — connected mesoscopic regions. This is how one quantifies
nanostructure in ionic liquids (polar vs apolar networks), microemulsions, and
block-copolymer domains without drawing an arbitrary isosurface.

Given labels $\ell_i\in\mathbb{Z}$ and the face-adjacency graph $G$ of the
tessellation, a domain is a connected component of the subgraph induced by a
fixed label. `voronoi_domains` returns per-label domain sizes and volumes:

```python
from molpy.compute import voronoi_domains

labels = np.arange(len(positions)) % 3  # per-atom integer label
domains = voronoi_domains(cells, labels)
```

Useful reductions:

- number of domains per label (fragmentation),
- volume fraction of the largest domain (percolation),
- mean domain volume (characteristic length $\sim V^{1/3}$).

---

## 3. Voids are the empty cells of the tessellation

Two common constructions:

1. **Probe atoms**: insert ghost sites at candidate free-volume locations (or use
   a fine auxiliary lattice), tessellate, and flag cells whose occupant is not a
   real atom.
2. **Boolean mask**: flag existing cells as void-like (e.g. very large $v_i$, or
   sites of removed particles) and aggregate connected void clusters.

`voronoi_voids` aggregates cells flagged by a per-cell boolean into void
clusters and volumes — free volume relevant to diffusion, gas solubility, and
porosity:

```python
from molpy.compute import voronoi_voids

is_void = np.zeros(len(positions), dtype=bool)  # per-cell bool
box_volume = box.volume
voids = voronoi_voids(cells, is_void, box_volume)
```

Void **volume fraction** $\phi_\mathrm{void}=V_\mathrm{void}/V_\mathrm{box}$ and
the size distribution of void clusters are the primary observables. Compare to
[local density](rdf.md) when you need a continuous field rather than a
discrete cell partition.

---

## 4. Voronoi integration: charges and molecular dipoles

### 4.1 Atomic charge from a density grid

Given a volumetric electron density $\rho_e(\mathbf{x})$ on a grid (e.g. from
AIMD / DFT) and nuclear charges $Z_i$, the Voronoi atomic charge is

$$
q_i = Z_i - \int_{V_i^\mathrm{rad} \rho_e(\mathbf{x})\,\mathrm{d}\mathbf{x}.
$$

The integral is a sum of voxel contributions whose centres (or fractional
volumes) lie in $V_i^\mathrm{rad}$. Charge neutrality of a neutral system is a
built-in sanity check: $\sum_i q_i \approx 0$ (residual reflects grid coarseness
and radii).

### 4.2 Molecular moments

With a map $\mathrm{atom}\to\mathrm{molecule}$, molecular charge and dipole are

$$
Q_m = \sum_{i\in m} q_i,
\qquad
\mathbf{M}_m
= \sum_{i\in m} q_i\,(\mathbf{r}_i - \mathbf{r}_m^\mathrm{ref}),
$$

where $\mathbf{r}_m^\mathrm{ref}$ is a molecular reference (COM or geometric
centre). These molecular dipoles are exactly what the dipole-flux route to
[infrared spectra](spectra.md) consumes — the bridge from an *ab initio* MD
electron-density trajectory to a predicted IR spectrum.

```python
from molpy.compute import DensityGrid, VoronoiIntegration

atomic_numbers = np.full(len(positions), 8, dtype=np.int64)
atom_to_mol = np.arange(len(positions), dtype=np.int64) // 4  # 4 atoms / mol
n_mol = int(atom_to_mol[-1]) + 1

dims = (16, 16, 16)
grid = DensityGrid(
    np.zeros(3),                              # origin (Å)
    (20.0 / 16) * np.eye(3),                  # voxel edge vectors (rows, Å)
    dims,
    rng.uniform(0.0, 0.1, size=16 * 16 * 16), # row-major densities
)

moments = VoronoiIntegration()(
    positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box
)
# moments -> per-molecule charges and dipole vectors
```

### 4.3 Grid resolution

Too coarse a density grid biases integrated charges (especially on light atoms).
Converge $\Delta x$ until $\sum_m Q_m$ and the IR spectrum stop changing. Report
grid spacing with any published dipole or IR intensity.

<figure id="fig-voronoi-vol" class="molcrafts-figure" markdown>
<div class="molcrafts-figure__body molcrafts-figure__body--chart">

```molplot preset="molplot" theme="auto" aspect="16:9"
mark:
  type: line
  strokeWidth: 2.2
  interpolate: monotone
data:
  values:
    - {v: 8, P: 0.1}
    - {v: 12, P: 0.4}
    - {v: 16, P: 1.2}
    - {v: 20, P: 1.5}
    - {v: 24, P: 0.9}
    - {v: 30, P: 0.3}
    - {v: 40, P: 0.05}
encoding:
  x:
    field: v
    type: quantitative
    title: cell volume (Å³)
  y:
    field: P
    type: quantitative
    scale: {zero: false}
    title: P(v)
  color:
    value: "#0284c7"
```

</div>

**Figure 1.** Schematic distribution of radical-Voronoi cell volumes in a dense liquid; the mean is $V/N$ for monodisperse equal radii.
</figure>

---

## 5. Pitfalls checklist

1. **Radii choice** → partition depends on $\{R_i\}$; use one consistent set and
   report it.
2. **Non-periodic box** → builder is periodic; supply the simulation box or
   surface cells are unbounded.
3. **Label/void array length** → `labels` and `is_void` must match cell count
   in tessellation order.
4. **Grid resolution for integration** → converge voxel size before trusting
   charges/dipoles.
5. **Charge neutrality** → large residual $\sum q_i$ signals grid or radius
   problems.
6. **Unwrapped vs wrapped positions** → tessellation needs wrapped (minimum-image
   consistent) coordinates inside the box used for the build.

---

## 6. References

- B. J. Gellatly, J. L. Finney, *J. Non-Cryst. Solids* **50**, 313 (1982) —
  radical (power) Voronoi tessellation.
- M. Thomas, M. Brehm, B. Kirchner, *Phys. Chem. Chem. Phys.* **17**, 3207
  (2015) — Voronoi integration of electron density for molecular dipoles.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — domains, voids, and AIMD analysis stack.

## See also

- [Spectra](spectra.md) — consumes Voronoi molecular dipoles.
- [Structural Analysis](rdf.md) — local density and packing.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
