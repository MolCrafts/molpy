"""
Polymer assembly module.

Provides linear polymer assembly with both topology-only and chemical reaction connectors,
plus optional geometric placement via Placer strategies.
"""

from .connectors import (
    Connector,
    ConnectorContext,
)
from .distributions import (
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    SchulzZimmPolydisperse,
    UniformPolydisperse,
)
from .growth_kernel import GrowthKernel, ProbabilityTableKernel
from .placer import CovalentSeparator, LinearOrienter, Placer, VdWSeparator
from .polymer_builder import PolymerBuilder, PolymerBuildResult
from .sequences import SequenceGenerator, WeightedSequenceGenerator
from .stochastic import (
    MonomerPlacement,
    MonomerTemplate,
    PortDescriptor,
    StochasticChain,
)
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
    # G-BigSMILES Stochastic Growth Types
    "MonomerTemplate",
    "PortDescriptor",
    "MonomerPlacement",
    "StochasticChain",
    # G-BigSMILES Growth Kernel
    "GrowthKernel",
    "ProbabilityTableKernel",
]
