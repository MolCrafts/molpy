"""Convenience exports for the builder subpackage.

The legacy ``PolymerBuilder`` class and bulk builders have been removed in
favour of the new declarative API documented in
``notebooks/reacter_polymerbuilder_integration.ipynb``.
"""

from .crystal import BlockRegion, CrystalBuilder, Lattice, Region, Site
from .polymer import *

__all__ = [
    "BlockRegion",
    "CrystalBuilder",
    "Lattice",
    "Region",
    "Site",
]
