"""System assembly — start here.

Polymer construction composes the real engine classes directly (there is
no ``polymer()`` dispatcher): prepare monomers with
:func:`molpy.parser.parse_monomer` + :func:`molpy.adapter.rdkit.generate_3d`,
mark the atoms that may react with ``fields.SITE``, then
:meth:`PolymerBuilder.build` a CGSmiles string. Crosslinking is the same
kernel with a different :class:`Selector`. Polydisperse systems drive
:class:`PolymerBuilder` from the distribution + :class:`SystemPlanner`
primitives. See :mod:`molpy.builder.assembly` for the full recipe.

Crystal construction goes through :func:`build_crystal` with
:class:`Lattice` / :class:`Site`. AmberTools-backed polymer builds use
:class:`AmberPolymerBuilder`.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from .ambertools import AmberResult, AmberTools
from .crystal import Lattice, Site, SpaceGroup, build_crystal
from .polymer import *
from .assembly import (
    Placer,
    PolymerBuilder,
    ResiduePlacer,
    ExhaustiveSelector,
    ExplicitPairSelector,
    GraphAssembler,
    MonomerLibrary,
    ProximitySelector,
    RandomSelector,
    Selector,
    SpacingSelector,
    TopologySelector,
)
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
    # Assembly: one kernel, one selector family
    "GraphAssembler",
    "PolymerBuilder",
    "Placer",
    "ResiduePlacer",
    "MonomerLibrary",
    "Selector",
    "TopologySelector",
    "ProximitySelector",
    "ExhaustiveSelector",
    "SpacingSelector",
    "ExplicitPairSelector",
    "RandomSelector",
]
