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
from .linear import linear
from .polydisperse import Polydisperse, SchulzZimm
from .sequence_generator import SequenceGenerator, WeightedSequenceGenerator
from .system import (
    Chain,
    DPDistribution,
    PolydisperseChainGenerator,
    SchulzZimmDPDistribution,
    SystemPlan,
    SystemPlanner,
)
from .types import ConnectionMetadata, ConnectionResult, PolymerBuildResult

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
    # Polydisperse (legacy)
    "Polydisperse",
    "SchulzZimm",
    # Sequence Generators
    "SequenceGenerator",
    "WeightedSequenceGenerator",
    # System-level (new three-layer architecture)
    "Chain",
    "DPDistribution",
    "PolydisperseChainGenerator",
    "SchulzZimmDPDistribution",
    "SystemPlan",
    "SystemPlanner",
    # Types
    "ConnectionMetadata",
    "ConnectionResult",
    "PolymerBuildResult",
]
