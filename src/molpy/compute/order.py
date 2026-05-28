"""Bond-orientational order operators — molrs-backed.

Thin ``Compute`` shells over ``molrs.compute.order.*``. Each forwards verbatim
to the Rust kernel and returns the molrs native result; molpy adds no wrapping
and copies nothing. Like ``compute.RDF``, these take two inputs, so the real
work lives in ``__call__`` and ``_compute`` raises.

References
----------
- Steinhardt, Nelson & Ronchetti, *Phys. Rev. B* **28**, 784 (1983).
- Nelson & Halperin, *Phys. Rev. B* **19**, 2457 (1979) (hexatic).
- ten Wolde, Ruiz-Montero & Frenkel, *J. Chem. Phys.* **104**, 9932 (1996)
  (solid-liquid).
- de Gennes & Prost, *The Physics of Liquid Crystals*, 2nd ed. (1993) (nematic).
"""

from __future__ import annotations

from collections.abc import Sequence

import molrs

from .base import Compute

_TWO_INPUT = "takes (frames, nlists); call the operator directly"


class Steinhardt(Compute):
    """Steinhardt :math:`q_\\ell` / :math:`w_\\ell` bond-orientational order.

    Parameters
    ----------
    l : Sequence[int]
        Spherical-harmonic degrees, e.g. ``[6]`` or ``[4, 6]``.
    average : bool
        Use the locally averaged variant.
    wl : bool
        Compute third-order :math:`w_\\ell` invariants instead of :math:`q_\\ell`.
    wl_normalize : bool
        Normalize the :math:`w_\\ell` invariants.
    """

    def __init__(
        self,
        l: Sequence[int],
        average: bool = False,
        wl: bool = False,
        wl_normalize: bool = False,
    ):
        super().__init__(l=l, average=average, wl=wl, wl_normalize=wl_normalize)
        self._inner = molrs.compute.order.Steinhardt(l, average, wl, wl_normalize)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError(f"Steinhardt {_TWO_INPUT}")


class Hexatic(Compute):
    """Two-dimensional :math:`\\psi_k` hexatic bond-orientational order.

    Parameters
    ----------
    k : int
        Symmetry order (``6`` for the hexatic order parameter).
    """

    def __init__(self, k: int):
        super().__init__(k=k)
        self._inner = molrs.compute.order.Hexatic(k)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError(f"Hexatic {_TWO_INPUT}")


class Nematic(Compute):
    """Nematic order parameter and Q-tensor from per-particle directors.

    The second call argument is a ``(N, 3)`` array of per-particle orientation
    vectors. Returns ``(order, eigenvalues, director, q_tensor)``.
    """

    def __init__(self):
        super().__init__()
        self._inner = molrs.compute.order.Nematic()

    def __call__(self, frames, directors):
        return self._inner.compute(frames, directors)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError("Nematic takes (frames, directors); call directly")


class SolidLiquid(Compute):
    """Solid-liquid classification via :math:`q_\\ell` bond correlations.

    Parameters
    ----------
    l : int
        Spherical-harmonic degree used for the bond correlation.
    q_threshold : float
        Minimum dot-product for a bond to count as solid-like.
    n_threshold : int
        Minimum number of solid-like bonds for a particle to be solid.
    """

    def __init__(self, l: int, q_threshold: float = 0.7, n_threshold: int = 6):
        super().__init__(l=l, q_threshold=q_threshold, n_threshold=n_threshold)
        self._inner = molrs.compute.order.SolidLiquid(l, q_threshold, n_threshold)

    def __call__(self, frames, nlists):
        return self._inner.compute(frames, nlists)

    def _compute(self, input):  # pragma: no cover — use __call__
        raise NotImplementedError(f"SolidLiquid {_TWO_INPUT}")
