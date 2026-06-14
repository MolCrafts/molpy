"""Local-environment observables — molrs-backed.

``BondOrder`` histograms neighbor bond directions on a (theta, phi) grid.
Thin shell over ``molrs.compute.environment.BondOrder``; takes
``(frames, nlists)`` like ``RDF``.
"""

from __future__ import annotations

import molrs

from .base import Compute


class BondOrder(Compute):
    """Bond-orientational order diagram on a spherical (theta, phi) grid.

    Parameters
    ----------
    n_theta : int
        Number of polar-angle bins.
    n_phi : int
        Number of azimuthal-angle bins.
    """

    def __init__(self, n_theta: int, n_phi: int):
        super().__init__(n_theta=n_theta, n_phi=n_phi)
        self._inner = molrs.compute.environment.BondOrder(n_theta, n_phi)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)
