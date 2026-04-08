"""Convenience exports for the builder subpackage.

The legacy ``PolymerBuilder`` class and bulk builders have been removed in
favour of the new declarative API documented in
``notebooks/reacter_polymerbuilder_integration.ipynb``.
"""

from .polymer.ambertools import AmberPolymerBuilder
from .crystal import BlockRegion, CrystalBuilder, Lattice, Region, Site
from .polymer import *

__all__ = [
    # AmberTools builders
    "AmberPolymerBuilder",
    # Crystal builders
    "BlockRegion",
    "CrystalBuilder",
    "Lattice",
    "Region",
    "Site",
]
