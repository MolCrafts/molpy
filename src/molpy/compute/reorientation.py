"""Legendre reorientational correlation functions — molrs-backed.

``LegendreReorientation`` computes the first- and second-order Legendre
reorientational time-correlation functions of bond (or molecular) vectors,

.. math::

    C_\\ell(t) = \\big\\langle P_\\ell\\big(\\mathbf u(0)\\cdot\\mathbf u(t)\\big)\\big\\rangle,

with ``P_1(x) = x`` and ``P_2(x) = (3x^2 - 1)/2``. ``C_2(t)`` is the quantity
probed by NMR and dielectric relaxation; its decay time is the reorientational
correlation time. Thin shell over the molrs TRAVIS-parity kernel; takes
``(frames, pairs)`` where ``pairs`` selects the vector endpoints.

References
----------
- B. J. Berne, R. Pecora, *Dynamic Light Scattering*, Wiley (1976) — reorientational
  correlation functions.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — TRAVIS.
"""

from __future__ import annotations

import molrs

from .base import Compute


class LegendreReorientation(Compute):
    """First/second Legendre reorientational TCFs ``C_1(t)``, ``C_2(t)``.

    Parameters
    ----------
    max_lag : int
        Longest lag (in frames) to evaluate.
    stride : int, default 1
        Stride between time origins.

    Notes
    -----
    Called as ``compute(frames, pairs)`` where ``pairs`` is an integer array of
    ``(i, j)`` endpoints defining each tracked vector. The result exposes
    ``lags``, ``c1``, and ``c2``.
    """

    def __init__(self, max_lag: int, stride: int = 1):
        super().__init__(max_lag=max_lag, stride=stride)
        self._inner = molrs.LegendreReorientation(max_lag, stride)

    def __call__(self, frames, pairs):
        return self._inner.compute(frames, pairs)
