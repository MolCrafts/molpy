"""Convenience exports for the builder subpackage.

Polymer construction goes through the declarative ``polymer()`` and
``polymer_system()`` entry points (see ``molpy.builder.polymer.dsl``);
``PolymerBuilder`` remains available for direct graph-based assembly.
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
