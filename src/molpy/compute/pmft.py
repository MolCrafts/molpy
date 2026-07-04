"""Potential of mean force and torque — molrs-backed.

``PMFTXY`` accumulates the 2-D potential of mean force in the local (x, y)
frame of each particle. Thin shell over ``molrs.compute.pmft.PMFTXY``; takes
``(frames, nlists)``. When the frames carry an ``orientations`` topology block,
every bond is rotated into each query particle's local frame (the per-particle
angle is ``atan2`` of its ``head - tail`` axis); otherwise it works in the lab
frame.

References
----------
- G. van Anders, D. Klotsa, N. K. Ahmed, M. Engel, S. C. Glotzer, *ACS Nano* **8**,
  931 (2014) — potential of mean force and torque.
- V. Ramasubramani et al., *Comput. Phys. Commun.* **254**, 107275 (2020) — the
  freud library, on which this kernel is modelled.
"""

from __future__ import annotations

import molrs

from .base import Compute


class PMFTXY(Compute):
    """2-D potential of mean force and torque on an (x, y) grid.

    Parameters
    ----------
    x_max, y_max : float
        Half-extent of the integration window along each local axis.
    n_x, n_y : int
        Grid resolution along each axis.
    """

    def __init__(self, x_max: float, y_max: float, n_x: int, n_y: int):
        super().__init__(x_max=x_max, y_max=y_max, n_x=n_x, n_y=n_y)
        self._inner = molrs.compute.pmft.PMFTXY(x_max, y_max, n_x, n_y)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)
