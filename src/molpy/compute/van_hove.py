"""Van Hove correlation function G(r, t) — molrs-backed.

``VanHove`` computes the self and distinct parts of the Van Hove correlation
function, the time-resolved generalization of the radial distribution function:
``G_s(r, t)`` is the probability that a particle has moved a distance ``r`` in
time ``t`` (it integrates to the self-diffusion picture), while ``G_d(r, t)``
tracks how the structure around a particle decorrelates. Thin shell over the
molrs analysis-parity kernel; takes ``(frames)``.

References
----------
- L. Van Hove, *Phys. Rev.* **95**, 249 (1954) — the correlation function G(r, t).
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

import molrs

from .base import Compute


class VanHove(Compute):
    """Van Hove correlation function ``G(r, t)`` (self + distinct parts).

    Parameters
    ----------
    n_rbins : int
        Number of radial bins.
    r_max : float
        Upper edge of the radial grid (Angstrom).
    lags : Sequence[int]
        Time lags (in frames) at which to evaluate ``G(r, t)``.
    stride : int, default 1
        Stride between time origins.

    Notes
    -----
    The result exposes ``r_centers``, ``lags``, ``g_self``, and ``g_distinct``
    (the latter only when ``has_distinct`` is true).
    """

    def __init__(
        self, n_rbins: int, r_max: float, lags: Sequence[int], stride: int = 1
    ):
        super().__init__(n_rbins=n_rbins, r_max=r_max, lags=lags, stride=stride)
        self._inner = molrs.VanHove(n_rbins, r_max, lags, stride)

    def __call__(self, frames):
        return self._inner.compute(frames)
