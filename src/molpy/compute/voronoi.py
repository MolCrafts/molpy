"""Radical (Laguerre) Voronoi tessellation, domains, voids & integration — molrs-backed.

The **radical** (power / Laguerre) Voronoi tessellation partitions space by
radius-weighted planes, so atoms of different size get cells proportional to
their radii — the physically correct partition for polydisperse systems. On top
of the tessellation:

- :func:`voronoi_domains` merges cells sharing a label into connected domains
  (e.g. polar vs. apolar nanostructuring in ionic liquids).
- :func:`voronoi_voids` aggregates the cells flagged as empty into void volumes.
- :class:`VoronoiIntegration` integrates an electron density over the cells to
  yield per-molecule charges and dipoles (Voronoi/atomic-charge partitioning),
  the basis for predicting infrared spectra from *ab initio* MD.

Thin shells over the molrs TRAVIS-parity kernels.

References
----------
- B. J. Gellatly, J. L. Finney, *J. Non-Cryst. Solids* **50**, 313 (1982) — radical
  (power) Voronoi tessellation.
- M. Thomas, M. Brehm, B. Kirchner, *Phys. Chem. Chem. Phys.* **17**, 3207 (2015)
  — Voronoi integration of the electron density for molecular dipoles.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — TRAVIS; domain and void analysis.
"""

from __future__ import annotations

import molrs

from .base import Compute

# Re-export the per-cell result type and the domain/void reductions.
VoronoiCells = molrs.VoronoiCells
voronoi_domains = molrs.voronoi_domains
voronoi_voids = molrs.voronoi_voids


class RadicalVoronoi(Compute):
    """Radical (Laguerre / power) Voronoi tessellation under periodic boundaries.

    Notes
    -----
    Called as ``compute(positions, radii, box)`` and returns
    :class:`VoronoiCells` (per-cell volumes, faces, and neighbours).
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.RadicalVoronoi()

    def __call__(self, positions, radii, box):
        return self._inner.build(positions, radii, box)


class VoronoiIntegration(Compute):
    """Integrate an electron density over radical-Voronoi cells.

    Aggregates a volumetric electron density into per-molecule charges and
    dipole moments, the Voronoi/atomic-charge partition used to derive infrared
    intensities from *ab initio* MD trajectories.

    Notes
    -----
    Called as ``compute(positions, radii, atomic_numbers, atom_to_mol, n_mol,
    grid, box)`` and returns the per-molecule moments.
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.VoronoiIntegration()

    def __call__(self, positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box):
        return self._inner.integrate(
            positions, radii, atomic_numbers, atom_to_mol, n_mol, grid, box
        )
