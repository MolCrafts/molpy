"""Polymer chains through AmberTools.

Private: every name here is public on :mod:`molpy.builder` only.

A GAFF chain is :class:`AmberPolymerBuilder`: prepgen cuts one
antechamber-typed oligomer and tleap ``sequence`` joins the residues, so that
path has no placer and no orienter. A port-joined chain — and a polydisperse
ensemble of them — is the native polymer model of :mod:`molpy.builder`
(``Fragment``, ``StochasticObject``, ``Molecule``, ``System``) sampled into a site
graph for :class:`molpy.builder.Assembler`.
"""

from molpy.builder.polymer.ambertools import (
    AmberBuildResult,
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
)

__all__ = [
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
]
