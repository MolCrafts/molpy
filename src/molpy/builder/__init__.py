"""System assembly — start here.

The native builders of :mod:`molrs.builder` are here by identity
(``mp.builder.Assembler is molrs.builder.Assembler``): site-graph assembly
(:class:`Assembler` with :class:`SitePlacer` + :class:`AxisOrienter`, or
:class:`GrowthPlacer` for a site graph without positions), coarse-graining
(:class:`Coarsener`, after :class:`molpy.SubgraphMatcher` finds the groups),
and the carbon nanostructures (:class:`GrapheneBuilder`,
:class:`CarbonTubeBuilder`, each building a :class:`~molpy.Frame`;
``mp.Atomistic.from_frame`` makes it a graph).

molpy adds:

* crystals — :class:`Lattice` / :class:`Site` / :class:`SpaceGroup`
  (``Lattice.build``; fractional ↔ Cartesian through :attr:`Lattice.box`);
* polymers — sequence generators, chain-length distributions and
  :class:`SystemPlanner`; :class:`AmberPolymerBuilder` for GAFF chains
  through AmberTools;
* virtual sites — :class:`VirtualSiteBuilder` with :class:`DrudeBuilder`
  (CL&Pol, parameters from ``mp.ff.params.clpol_polarizability``) and
  :class:`Tip4pBuilder`;
* :class:`PackingTemplate` — a built molecule's frame plus its hydrogen
  indices, for a packer.

Every name has this one path; the modules behind it are private.
"""

from molrs.builder import *  # noqa: F403
from molrs.builder import __all__ as _native

from ._crystal import Lattice, Site
from ._packing import PackingTemplate
from ._polymer import (
    AlternatingSequenceGenerator,
    AmberBuildResult,
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
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
from ._symmetry import SpaceGroup
from ._virtualsite import DrudeBuilder, Tip4pBuilder, VirtualSiteBuilder

__all__ = [
    # Native builders (molrs.builder)
    *_native,
    # Crystal builders
    "Lattice",
    "Site",
    "SpaceGroup",
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
    # GAFF polymers through AmberTools
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
    # Virtual-site augmentation
    "VirtualSiteBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    # Packing input
    "PackingTemplate",
]
