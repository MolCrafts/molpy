"""System assembly — start here.

Polymer and backmap construction compose native primitives, re-exported on
the ``molpy`` root: coarse-grain a structure into a site graph
(:class:`molpy.SubgraphMatcher`, :class:`molpy.Coarsener`), then build one
template copy per site with :class:`molpy.Assembler`
(:class:`molpy.SitePlacer` + :class:`molpy.AxisOrienter`). Polydisperse
systems plan chains with the distribution + :class:`SystemPlanner`
primitives here.

Crystal construction goes through :meth:`Lattice.build` with
:class:`Lattice` / :class:`Site`. Nanostructures expose direct ``build``
methods; their compile/cache details remain internal.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from ._finalize import Finalization, StructureFinalizer
from .ambertools import AmberResult, AmberTools
from .crystal import Lattice, Site, SpaceGroup
from .nanostructure import CarbonTubeBuilder, GrapheneBuilder
from .polymer import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    Chain,
    DPDistribution,
    FlorySchulzPolydisperse,
    MassDistribution,
    PoissonPolydisperse,
    PolydisperseChainGenerator,
    SchulzZimmPolydisperse,
    SequenceGenerator,
    SystemPlan,
    SystemPlanner,
    UniformPolydisperse,
    WeightedSequenceGenerator,
)
from .virtualsite import (
    DrudeBuilder,
    Tip4pBuilder,
    VirtualSiteBuilder,
    load_polarizability,
)

__all__ = [
    # AmberTools
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
    # Nanostructure builders
    "CarbonTubeBuilder",
    "GrapheneBuilder",
    # Polymer planning primitives
    "AlternatingSequenceGenerator",
    "BlockSequenceGenerator",
    "Chain",
    "DPDistribution",
    "FlorySchulzPolydisperse",
    "MassDistribution",
    "PoissonPolydisperse",
    "PolydisperseChainGenerator",
    "SchulzZimmPolydisperse",
    "SequenceGenerator",
    "SystemPlan",
    "SystemPlanner",
    "UniformPolydisperse",
    "WeightedSequenceGenerator",
    # Virtual-site augmentation
    "VirtualSiteBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    "load_polarizability",
    # Finalization
    "StructureFinalizer",
    "Finalization",
]
