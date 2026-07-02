"""Spatial distribution function (SDF) — molrs-backed.

``SpatialDistribution`` accumulates the density of *target* atoms on a body-fixed
3-D grid, after Kabsch-aligning each reference molecule to a template geometry.
This is the orientation-resolved generalization of the radial distribution
function: it reveals *where* around a molecule its neighbours sit, not just how
far. Thin shell over the molrs analysis-parity kernel; takes ``(frames)``.

References
----------
- I. M. Svishchev, P. G. Kusalik, *J. Chem. Phys.* **99**, 3049 (1993);
  P. G. Kusalik, I. M. Svishchev, *Science* **265**, 1219 (1994) — spatial
  distribution functions.
- M. Brehm, B. Kirchner, *J. Chem. Inf. Model.* **51**, 2007 (2011) — reference implementation SDF.
"""

from __future__ import annotations

from collections.abc import Sequence

import molrs

from .base import Compute


class SpatialDistribution(Compute):
    """Target-atom density on a molecule body-fixed grid (SDF).

    Parameters
    ----------
    reference : Sequence[int]
        Atom indices whose instances define the body-fixed frame each step.
    template : ndarray
        ``(n_ref, 3)`` reference geometry the per-frame reference atoms are
        Kabsch-aligned to.
    target : Sequence[int]
        Atom indices whose density is accumulated on the grid.
    n : tuple[int, int, int]
        Grid resolution along each body-fixed axis.
    extent : tuple[float, float, float]
        Half-extent (Angstrom) of the grid along each axis.
    bulk_density : float, optional
        If given, the result also exposes ``g_sdf`` (the density normalized by
        the bulk number density).
    orientation_pairs : ndarray, optional
        Atom-index pairs whose mean orientation is accumulated per voxel.
    """

    def __init__(
        self,
        reference: Sequence[int],
        template,
        target: Sequence[int],
        n: tuple[int, int, int],
        extent: tuple[float, float, float],
        bulk_density: float | None = None,
        orientation_pairs=None,
    ):
        super().__init__(
            reference=reference,
            template=template,
            target=target,
            n=n,
            extent=extent,
            bulk_density=bulk_density,
            orientation_pairs=orientation_pairs,
        )
        self._inner = molrs.SpatialDistribution(
            reference, template, target, n, extent, bulk_density, orientation_pairs
        )

    def __call__(self, frames):
        return self._inner.compute(frames)
