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
]
