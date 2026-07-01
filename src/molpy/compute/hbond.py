"""Hydrogen-bond detection — molrs-backed.

``HBonds`` detects geometric hydrogen bonds per frame from explicit donor
``(D, H)`` pairs and acceptor atoms, using the :class:`HBondCriterion`
(donor-acceptor distance and D-H...A angle). Pair these counts with the
:class:`~molpy.compute.persist.Persist` survival analysis for hydrogen-bond
lifetime dynamics. Thin shell over the molrs TRAVIS-parity kernel; takes
``(frames)``.

References
----------
- A. Luzar, D. Chandler, *Nature* **379**, 55 (1996); *Phys. Rev. Lett.* **76**,
  928 (1996) — geometric hydrogen-bond criterion and bond-lifetime kinetics.
- M. Brehm, M. Thomas, S. Gehrke, B. Kirchner, *J. Chem. Phys.* **152**, 164105
  (2020) — TRAVIS.
"""

from __future__ import annotations

import molrs

from .base import Compute

# Re-export the configuration object so callers can tune the geometric criterion.
HBondCriterion = molrs.HBondCriterion


class HBonds(Compute):
    """Detect hydrogen bonds per frame from explicit donors and acceptors.

    Parameters
    ----------
    donors : ndarray
        ``(n_donor, 2)`` integer array of ``(D, H)`` atom-index pairs.
    acceptors : ndarray
        Integer array of acceptor atom indices.
    criterion : HBondCriterion, optional
        Geometric criterion; defaults to Luzar-Chandler values
        (3.5 Angstrom donor-acceptor cutoff, 150 deg angle cutoff).

    Notes
    -----
    The result exposes ``per_frame`` (lists of ``(D, H, A, distance, angle)``)
    and ``counts`` (hydrogen bonds per frame).
    """

    def __init__(self, donors, acceptors, criterion=None):
        super().__init__(donors=donors, acceptors=acceptors, criterion=criterion)
        self._inner = molrs.HBonds(donors, acceptors, criterion)

    def __call__(self, frames):
        return self._inner.compute(frames)
