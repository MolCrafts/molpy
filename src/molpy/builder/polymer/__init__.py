"""
Polymer assembly module.

Provides linear polymer assembly with both topology-only and chemical reaction connectors,
plus optional geometric placement via Placer strategies.
"""

from .connectors import (
    AutoConnector,
    BondKind,
    CallbackConnector,
    ChainConnector,
    Connector,
    ConnectorContext,
    ReacterConnector,
    TableConnector,
    TopologyConnector,
)
from .growth_kernel import GrowthKernel, ProbabilityTableKernel
from .linear import linear
from .polymer_builder import PolymerBuilder
from .polydisperse import Polydisperse, SchulzZimm
from .sequence_generator import SequenceGenerator, WeightedSequenceGenerator
from .stochastic_generator import StochasticChainGenerator
from .system import (
    Chain,
    FlorySchulzPolydisperse,
    PoissonPolydisperse,
    PolydisperseChainGenerator,
    SchulzZimmPolydisperse,
    SystemPlan,
    SystemPlanner,
    UniformPolydisperse,
)
from .types import (
    ConnectionMetadata,
    ConnectionResult,
    MonomerPlacement,
    MonomerTemplate,
    PolymerBuildResult,
    PortDescriptor,
    StochasticChain,
)

__all__ = [
    "AutoConnector",
    "BondKind",
    "CallbackConnector",
    "ChainConnector",
    # Connectors
    "Connector",
    "ConnectorContext",
    "ReacterConnector",
    "TableConnector",
    "TopologyConnector",
    "linear",
    # CGSmiles Builder
    "PolymerBuilder",
    # Polydisperse (legacy)
    "Polydisperse",
    "SchulzZimm",
    # Sequence Generators
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # System-level (new three-layer architecture)
    "Chain",
    "FlorySchulzPolydisperse",
    "PoissonPolydisperse",
    "PolydisperseChainGenerator",
    "SchulzZimmPolydisperse",
    "SystemPlan",
    "SystemPlanner",
    "UniformPolydisperse",
    # Types
    "ConnectionMetadata",
    "ConnectionResult",
    "PolymerBuildResult",
    # G-BigSMILES Stochastic Growth Types
    "MonomerTemplate",
    "PortDescriptor",
    "MonomerPlacement",
    "StochasticChain",
    # G-BigSMILES Growth Kernel
    "GrowthKernel",
    "ProbabilityTableKernel",
    # G-BigSMILES Stochastic Generator
    "StochasticChainGenerator",
]
