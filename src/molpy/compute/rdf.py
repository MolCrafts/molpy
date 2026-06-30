"""Radial distribution function g(r) — molrs-backed.

Returns ``molrs.compute.density.RDFResult`` directly (no molpy wrapper).
The result is finalized eagerly inside ``RDF.compute``, so ``result.rdf``
is the normalized g(r) and not the raw histogram.

References
----------
- V. Ramasubramani, B. D. Dice, E. S. Harper, M. P. Spellings, J. A. Anderson,
  S. C. Glotzer, *Comput. Phys. Commun.* **254**, 107275 (2020) — the freud
  library, on which this kernel is modelled.
- M. P. Allen, D. J. Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
  Oxford (2017) — g(r) and coordination numbers.
"""

from __future__ import annotations

from molrs.compute.density import RDF as _MolrsRDF
from molrs.compute.density import RDFResult as _MolrsRDFResult

from .base import Compute


class RDF(Compute):
    """Histogram pair distances into g(r) over one or more frames.

    Parameters
    ----------
    n_bins : int
        Number of histogram bins.
    r_max : float
        Upper edge of the last bin in Angstroms.
    r_min : float, default 0.0
        Lower edge of bin 0 in Angstroms.

    Notes
    -----
    RDF takes two data inputs (frames + neighbor lists).
    """

    def __init__(self, n_bins: int, r_max: float, r_min: float = 0.0):
        super().__init__(n_bins=n_bins, r_max=r_max, r_min=r_min)
        self._rdf = _MolrsRDF(n_bins, r_max, r_min)

    def __call__(self, frames, neighbors) -> _MolrsRDFResult:
        frame_list = self._as_list(frames)
        for f in frame_list:
            if f.box.is_free:
                raise ValueError("frame.box is required for RDF computation")
        # molpy.Frame IS-A molrs.Frame — passed through PyO3 downcast
        # with no conversion / column copy.
        return self._rdf.compute(frame_list, self._as_list(neighbors))

    @staticmethod
    def _as_list(x):
        if isinstance(x, (list, tuple)):
            return list(x)
        return [x]
