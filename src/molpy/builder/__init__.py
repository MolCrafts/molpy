"""System assembly — start here.

Polymer and backmap construction compose native primitives, re-exported on
the ``molpy`` root: coarse-grain a structure into a site graph
(:class:`molpy.SubgraphMatcher`, :class:`molpy.Coarsener`), then build one
template copy per site with :class:`molpy.Assembler`
(:class:`molpy.SitePlacer` + :class:`molpy.AxisOrienter`). Polydisperse
systems plan chains with the distribution + :class:`SystemPlanner`
primitives here.

A built molecule goes to a packer through :class:`PackingTemplate`: its
frame plus the indices of its hydrogens.

Crystal construction goes through :meth:`Lattice.build` with
:class:`Lattice` / :class:`Site`. Nanostructures expose direct ``build``
methods; their compile/cache details remain internal.
"""

from ._finalize import Finalization, StructureFinalizer
from .crystal import Lattice, Site, SpaceGroup
from .nanostructure import CarbonTubeBuilder, GrapheneBuilder
from .packing import PackingTemplate
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
    # Crystal builders
    "Lattice",
    "Site",
    "SpaceGroup",
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
    # Packing input
    "PackingTemplate",
    # Finalization
    "StructureFinalizer",
    "Finalization",
]
