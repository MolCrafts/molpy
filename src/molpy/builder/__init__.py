"""Convenience exports for the builder subpackage.

The legacy ``PolymerBuilder`` class and bulk builders have been removed in
favour of the new declarative API documented in
``notebooks/reacter_polymerbuilder_integration.ipynb``.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from .crystal import Lattice, Site, build_crystal
from .polymer import *
from .polymer.ambertools import AmberPolymerBuilder

__all__ = [
    # AmberTools builders
    "AmberPolymerBuilder",
    # Crystal builders
    "BoxRegion",
    "Cube",
    "Lattice",
    "Region",
    "Site",
    "SphereRegion",
    "build_crystal",
]
