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
:class:`AmberPolymerBuilder`. Nanostructures expose direct ``build`` methods;
their compile/cache details remain internal.
"""

from molpy.core.region import BoxRegion, Cube, Region, SphereRegion

from ._finalize import Finalization, StructureFinalizer
from .ambertools import AmberResult, AmberTools
from .crystal import Lattice, Site, SpaceGroup, build_crystal
from .nanostructure import CarbonTubeBuilder
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
from .assembly import (
    AssemblyFinalizer,
    Placer,
    PolymerBuilder,
    ResiduePlacer,
    ExhaustiveSelector,
    ExplicitPairSelector,
    GraphAssembler,
    MonomerLibrary,
    ProximitySelector,
    RandomSelector,
    Replicas,
    Selector,
    SiteMap,
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
    # Nanostructure builders
    "CarbonTubeBuilder",
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
    # Assembly: one kernel, one selector family
    "GraphAssembler",
    "AssemblyFinalizer",
    "StructureFinalizer",
    "Finalization",
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
    "SiteMap",
    "Replicas",
]
