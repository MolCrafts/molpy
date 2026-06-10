"""
Polymer assembly module.

Provides linear polymer assembly with both topology-only and chemical reaction connectors,
plus optional geometric placement via Placer strategies.
"""

from .connectors import (
    Connector,
    ConnectorContext,
)
from .dsl import (
    BuildPolymer,
    BuildPolymerAmber,
    BuildSystem,
    PlanSystem,
    PrepareMonomer,
    generate_3d,
    polymer,
    polymer_system,
    prepare_monomer,
)
from .distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
)
from .core import PolymerBuilder, PolymerBuildResult
from .placer import CovalentSeparator, LinearOrienter, Placer, VdWSeparator
from .sequences import SequenceGenerator, WeightedSequenceGenerator
from .system import (
    Chain,
    PolydisperseChainGenerator,
    SystemPlan,
    SystemPlanner,
)

__all__ = [
    # Connector
    "Connector",
    "ConnectorContext",
    # Placer
    "CovalentSeparator",
    "LinearOrienter",
    "Placer",
    "VdWSeparator",
    # CGSmiles Builder
    "PolymerBuilder",
    "PolymerBuildResult",
    # Sequence Generators
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # Distributions
    "DPDistribution",
    "MassDistribution",
    "FlorySchulzPolydisperse",
    "PoissonPolydisperse",
    "SchulzZimmPolydisperse",
    "UniformPolydisperse",
    # System-level (three-layer architecture)
    "Chain",
    "PolydisperseChainGenerator",
    "SystemPlan",
    "SystemPlanner",
    # DSL tools and entry functions
    "PrepareMonomer",
    "BuildPolymer",
    "PlanSystem",
    "BuildSystem",
    "BuildPolymerAmber",
    "polymer",
    "polymer_system",
    "prepare_monomer",
    "generate_3d",
]
