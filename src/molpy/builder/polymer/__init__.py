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

from .polymer_builder import PolymerBuilder
from .sequence_generator import SequenceGenerator, WeightedSequenceGenerator
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
    # CGSmiles Builder
    "PolymerBuilder",
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
]
