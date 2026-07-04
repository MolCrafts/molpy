# Radical Voronoi: Domains, Voids & Charge Integration

This page is a self-contained, textbook-style introduction to the **radical
Voronoi tessellation** MolPy ports from the reference implementation, and the three analyses built on
it: connected-domain detection, void analysis, and electron-density integration
into per-molecule charges and dipoles. A Voronoi tessellation assigns every point
in space to its nearest atom; the **radical** (Laguerre / power) variant weights
the partition by atomic radii, which is the physically correct division for
systems of differently-sized atoms.

The tessellation and reductions run in Rust (`molrs`); they operate directly on
positions, radii, and the box rather than on `Frame` objects.

!!! note "Conventions used throughout"
    - Positions and radii are in Å; volumes in Å³.
    - The tessellation is fully periodic (minimum image).
    - A *cell* is one atom's radical-Voronoi polyhedron; *domains* and *voids* are
      groupings of cells.

---

## 1. The radical tessellation partitions space by power distance

In an ordinary Voronoi tessellation, the boundary between two atoms is the
perpendicular bisector of their separation — correct only when all atoms are the
same size. The **radical** tessellation instead places the boundary where the
*power distances* are equal,

$$
\pi_i(\mathbf r) = |\mathbf r - \mathbf r_i|^2 - R_i^2,
$$

so a larger atom (bigger $R_i$) claims a proportionally larger cell. Each atom's
cell volume is a meaningful local volume — the basis for local density, packing
fraction, and the analyses below.

```python
from molpy.compute import RadicalVoronoi

cells = RadicalVoronoi()(positions, radii, box)   # -> VoronoiCells
cells.neighbors(i)    # cells sharing a face with cell i
```

---

## 2. Domains group cells that belong together

Labelling each atom (by species, charge sign, polarity, …) and merging
face-sharing cells with the same label yields **domains** — the connected
mesoscopic regions that characterize nanostructured liquids, such as the polar and
apolar networks of ionic liquids. `voronoi_domains` returns the per-label domain
sizes and volumes:

```python
from molpy.compute import voronoi_domains

domains = voronoi_domains(cells, labels)   # labels: per-atom integer label
```

---

## 3. Voids are the empty cells of the tessellation

Flagging cells that contain no "real" occupant and aggregating their connected
clusters maps the **void** space — cavities and free volume relevant to diffusion,
gas solubility, and porosity:

```python
from molpy.compute import voronoi_voids

voids = voronoi_voids(cells, is_void, box_volume)   # is_void: per-cell bool
```

---

## 4. Voronoi integration yields molecular charges and dipoles

Integrating a volumetric **electron density** over each radical-Voronoi cell and
summing per molecule partitions the total charge into atomic/molecular
contributions — the Voronoi route to per-molecule charges and **dipole moments**.
These molecular dipoles are precisely what the dipole-autocorrelation route to
[infrared spectra](spectra.md) consumes, which is why this analysis is the bridge
from an *ab initio* MD electron density (e.g. a cube trajectory) to a predicted IR
spectrum.

```python
from molpy.compute import VoronoiIntegration

moments = VoronoiIntegration()(
    positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box
)
# moments -> per-molecule charges and dipole vectors
```

---

## 5. Pitfalls checklist

1. **Radii choice** → the partition depends on the atomic radii; use a consistent,
   physically motivated set (e.g. van der Waals or covalent) and report it.
2. **Non-periodic box** → the builder is periodic; supply the simulation box, not a
   free frame, or surface cells are unbounded.
3. **Label/void arrays length** → `labels` and `is_void` must have one entry per
   atom/cell, in tessellation order.
4. **Grid resolution for integration** → too coarse a density grid biases the
   integrated charges; converge the grid spacing.
5. **Charge neutrality** → integrated per-molecule charges should sum to the system
   charge; a large residual signals a grid or radius problem.

---

## 6. References

- B. J. Gellatly, J. L. Finney, *J. Non-Cryst. Solids* **50**, 313 (1982) — radical
  (power) Voronoi tessellation.
- M. Thomas, M. Brehm, B. Kirchner, *Phys. Chem. Chem. Phys.* **17**, 3207 (2015)
  — Voronoi integration of the electron density for molecular dipoles.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation; domain and void analysis.

## See also

- [Vibrational Spectra from MD](spectra.md) — consumes the Voronoi molecular dipoles.
- [Structural Analysis](structure.md) — local density and packing.
- [Compute overview](index.md) — the Compute → Result pattern.
- [API reference: Compute](../api/compute.md).
