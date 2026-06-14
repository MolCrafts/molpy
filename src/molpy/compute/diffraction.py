"""Diffraction observables — molrs-backed.

``StaticStructureFactorDebye`` computes the static structure factor

.. math::

    S(k) = \\frac{1}{N} \\left\\langle \\sum_{i,j}
            \\frac{\\sin(k r_{ij})}{k r_{ij}} \\right\\rangle

via the Debye scattering equation. Thin shell over
``molrs.compute.diffraction.StaticStructureFactorDebye``; takes ``(frames)``.
"""

from __future__ import annotations

import molrs

from .base import Compute


class StaticStructureFactorDebye(Compute):
    """Static structure factor S(k) via the Debye equation.

    Parameters
    ----------
    k_values : array-like
        Scattering wavenumbers (1/Angstrom) at which to evaluate S(k).
    """

    def __init__(self, k_values):
        super().__init__(k_values=k_values)
        self._inner = molrs.compute.diffraction.StaticStructureFactorDebye(k_values)

    def __call__(self, frames):
        return self._inner.compute(frames)
