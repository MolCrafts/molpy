"""Polymer sub-primitives: sequences, chain-length distributions, system plans.

Private: every name here is public on :mod:`molpy.builder` only.

A port-joined chain is :class:`molpy.builder.Assembler` over a site graph. A GAFF
chain is :class:`AmberPolymerBuilder`: prepgen cuts one antechamber-typed
oligomer and tleap ``sequence`` joins the residues, so that path has no
placer and no orienter. What remains here besides those two is how to pick
the next monomer label (:mod:`sequences`), how long the chains are
(:mod:`distributions`), and how many of each to make (:mod:`system`).
"""

from molpy.builder.polymer.ambertools import (
    AmberBuildResult,
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
)
from molpy.builder.polymer.distributions import (
    DpDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
)
from molpy.builder.polymer.sequences import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    SequenceGenerator,
    WeightedSequenceGenerator,
)
from molpy.builder.polymer.system import (
    Chain,
    PolydisperseChainGenerator,
    SystemPlan,
    SystemPlanner,
)

__all__ = [
    # Amber chain: tleap sequence, no placer
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
    # Sequence generators
    "AlternatingSequenceGenerator",
    "BlockSequenceGenerator",
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # Distributions
    "DpDistribution",
    "MassDistribution",
    "FlorySchulzPolydisperse",
    "PoissonPolydisperse",
    "SchulzZimmPolydisperse",
    "UniformPolydisperse",
    # System planning
    "Chain",
    "SystemPlan",
    "SystemPlanner",
    "PolydisperseChainGenerator",
]
