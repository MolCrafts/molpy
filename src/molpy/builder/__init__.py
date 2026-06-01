"""Convenience exports for the builder subpackage.

The legacy ``PolymerBuilder`` class and bulk builders have been removed in
favour of the new declarative API documented in
``notebooks/reacter_polymerbuilder_integration.ipynb``.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from .crystal import Lattice, Site, build_crystal
from .polymer import *
from .polymer.ambertools import AmberPolymerBuilder
from .polymer.dsl import (
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
    # Polymer DSL tools and entry functions
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
