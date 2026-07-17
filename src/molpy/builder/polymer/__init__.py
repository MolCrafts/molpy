"""Polymer sub-primitives: sequences, chain-length distributions, system plans.

Chain assembly itself lives in :mod:`molpy.builder.assembly`
(:class:`~molpy.builder.assembly.PolymerBuilder` +
:class:`~molpy.builder.assembly.MonomerLibrary`). What remains here is what a
polymer needs *besides* the assembler: how to pick the next monomer label
(:mod:`sequences`), how long the chains are (:mod:`distributions`), and how many
of each to make (:mod:`system`).
"""

from .distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
)
from .sequences import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    SequenceGenerator,
    WeightedSequenceGenerator,
)
from .system import (
    Chain,
    PolydisperseChainGenerator,
    SystemPlan,
    SystemPlanner,
)

__all__ = [
    # Placer
    # Sequence generators
    "AlternatingSequenceGenerator",
    "BlockSequenceGenerator",
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # Distributions
    "DPDistribution",
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
