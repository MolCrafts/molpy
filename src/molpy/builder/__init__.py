"""System assembly — start here.

Polymer construction goes through the declarative entry functions:

- :func:`polymer` — build a single chain in one call
- :func:`polymer_system` — build a polydisperse multi-chain system
- :func:`prepare_monomer` — BigSMILES → 3D monomer with ports

Advanced, step-by-step assembly lives in :mod:`molpy.builder.polymer`
(``PolymerBuilder``, ``Connector``, placers, ``ReactionPresets``).
Crystal construction goes through :func:`build_crystal` with
:class:`Lattice` / :class:`Site`. AmberTools-backed polymer builds use
:class:`AmberPolymerBuilder` (or ``polymer(..., backend="amber")``).
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from .crystal import Lattice, Site, SpaceGroup, build_crystal
from .polymer import *
from .polymer.ambertools import AmberPolymerBuilder
from .polymer.dsl import (
    generate_3d,
    polymer,
    polymer_system,
    prepare_monomer,
)
from .virtualsite import (
    DrudeBuilder,
    Tip4pBuilder,
    VirtualSiteBuilder,
    load_polarizability,
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
    "SpaceGroup",
    "SphereRegion",
    "build_crystal",
    # Polymer entry functions
    "polymer",
    "polymer_system",
    "prepare_monomer",
    "generate_3d",
    # Virtual-site augmentation
    "VirtualSiteBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    "load_polarizability",
]
