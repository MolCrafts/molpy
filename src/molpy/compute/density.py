"""Density-field operators — molrs-backed.

Thin ``Compute`` shells over ``molrs.compute.density.*``, returning the molrs
native results unchanged. ``LocalDensity`` takes ``(frames, nlists)`` (like
``RDF``); ``GaussianDensity`` takes ``(frames)`` only.

References
----------
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud library, on which these density kernels are modelled.
"""

from __future__ import annotations

import molrs

from .base import Compute


class LocalDensity(Compute):
    """Per-particle local number density within a cutoff sphere.

    Parameters
    ----------
    r_max : float
        Cutoff radius (Angstrom) of the counting sphere.
    diameter : float
        Particle diameter correction; ``0.0`` counts particle centres only.
    """

    def __init__(self, r_max: float, diameter: float = 0.0):
        super().__init__(r_max=r_max, diameter=diameter)
        self._inner = molrs.compute.density.LocalDensity(r_max, diameter)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)


class GaussianDensity(Compute):
    """Gaussian-smeared number density on a 3-D grid.

    Parameters
    ----------
    nx, ny, nz : int
        Grid resolution along each axis.
    sigma : float
        Gaussian smearing width (Angstrom).
    """

    def __init__(self, nx: int, ny: int, nz: int, sigma: float):
        super().__init__(nx=nx, ny=ny, nz=nz, sigma=sigma)
        self._inner = molrs.compute.density.GaussianDensity(nx, ny, nz, sigma)

    def __call__(self, frames):
        return self._inner.compute(frames)
