"""Geometric distribution functions (ADF / DDF / distance-DF, CDF) — molrs-backed.

Thin ``Compute`` shells over ``molrs`` analysis-parity geometric distributions.
Each forwards verbatim to the Rust kernel and returns the molrs native result.
Each takes ``(frames)`` only. The atom tuples to histogram are read from each
frame's core topology blocks — ``bonds`` (pairs) for distances, ``angles``
(triplets) for angles, ``dihedrals`` (quadruplets) for dihedrals — so no separate
group-index array is passed.

References
----------
- M. Brehm, B. Kirchner, *J. Chem. Inf. Model.* **51**, 2007 (2011) — reference implementation;
  radial/angular/dihedral and combined distribution functions.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — reference implementation, current feature set.
"""

from __future__ import annotations

from collections.abc import Sequence

import molrs

from .base import Compute


class DistanceDistribution(Compute):
    """Distance distribution function over ``(i, j)`` atom pairs.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    min, max : float
        Lower and upper distance edges (Angstrom).
    """

    def __init__(self, n_bins: int, min: float, max: float):
        super().__init__(n_bins=n_bins, min=min, max=max)
        self._inner = molrs.DistanceDistribution(n_bins, min, max)

    def __call__(self, frames):
        return self._inner.compute(frames)


class AngleDistribution(Compute):
    """Angular distribution function (ADF) over ``(i, j, k)`` triplets.

    The angle is taken at the middle atom ``j``. The result's
    ``density_sin_corrected`` removes the trivial ``sin(theta)`` solid-angle
    weighting so a structureless distribution is flat.

    Parameters
    ----------
    n_bins : int
        Number of angular bins.
    min, max : float, default 0.0 / 180.0
        Angle range in degrees.
    """

    def __init__(self, n_bins: int, min: float = 0.0, max: float = 180.0):
        super().__init__(n_bins=n_bins, min=min, max=max)
        self._inner = molrs.AngleDistribution(n_bins, min, max)

    def __call__(self, frames):
        return self._inner.compute(frames)


class DihedralDistribution(Compute):
    """Dihedral distribution function (DDF) over ``(i, j, k, l)`` quadruplets.

    Parameters
    ----------
    n_bins : int
        Number of angular bins.
    min, max : float, default -180.0 / 180.0
        Dihedral range in degrees.
    """

    def __init__(self, n_bins: int, min: float = -180.0, max: float = 180.0):
        super().__init__(n_bins=n_bins, min=min, max=max)
        self._inner = molrs.DihedralDistribution(n_bins, min, max)

    def __call__(self, frames):
        return self._inner.compute(frames)


class CombinedDistribution(Compute):
    """Joint (combined) distribution over several geometric observables — the
    reference implementation combined distribution function (CDF).

    Parameters
    ----------
    axes : Sequence[tuple[str, int, float, float, bool]]
        One ``(kind, n_bins, min, max, sin_weight)`` per axis, where ``kind`` is
        ``"distance"``, ``"angle"``, or ``"dihedral"``.

    Notes
    -----
    Called as ``compute(frames)``; each axis reads its atom tuples from the
    matching topology block (``bonds`` / ``angles`` / ``dihedrals``) of every
    frame.
    """

    def __init__(self, axes: Sequence[tuple[str, int, float, float, bool]]):
        super().__init__(axes=axes)
        self._inner = molrs.CombinedDistribution(axes)

    def __call__(self, frames):
        return self._inner.compute(frames)
