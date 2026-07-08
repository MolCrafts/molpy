"""System assembly — start here.

Polymer construction composes the real engine classes directly (there is
no ``polymer()`` dispatcher): prepare monomers with
:func:`molpy.parser.parse_monomer` + :func:`molpy.adapter.rdkit.generate_3d`,
then assemble with :class:`PolymerBuilder` (``.build_sequence`` or
``.build`` on a CGSmiles string). Polydisperse systems drive
:class:`PolymerBuilder` from the distribution + :class:`SystemPlanner`
primitives. See :mod:`molpy.builder.polymer` for the full recipe.

Crystal construction goes through :func:`build_crystal` with
:class:`Lattice` / :class:`Site`. AmberTools-backed polymer builds use
:class:`AmberPolymerBuilder`.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from .ambertools import AmberResult, AmberTools
from .crosslink import Crosslinker, DeterministicCrosslinker
from .crystal import Lattice, Site, SpaceGroup, build_crystal
from .polymer import *
from .polymer.ambertools import AmberPolymerBuilder
from .virtualsite import (
    DrudeBuilder,
    Tip4pBuilder,
    VirtualSiteBuilder,
    load_polarizability,
)

__all__ = [
    # AmberTools builders
    "AmberPolymerBuilder",
    "AmberTools",
    "AmberResult",
    # Crystal builders
    "BoxRegion",
    "Cube",
    "Lattice",
    "Region",
    "Site",
    "SpaceGroup",
    "SphereRegion",
    "build_crystal",
    # Virtual-site augmentation
    "VirtualSiteBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    "load_polarizability",
    # Offline crosslinking
    "Crosslinker",
    "DeterministicCrosslinker",
]
