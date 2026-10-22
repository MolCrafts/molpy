"""System assembly — start here.

The native builders of :mod:`molrs.builder` are here by identity
(``mp.builder.Assembler is molrs.builder.Assembler``): site-graph assembly
(:class:`Assembler` with :class:`SitePlacer` + :class:`AxisOrienter`, or
:class:`GrowthPlacer` for a site graph without positions), coarse-graining
(:class:`Coarsener`, after :class:`molpy.perceive.SubgraphMatcher` finds the groups),
and the carbon nanostructures (:class:`GrapheneBuilder`,
:class:`CarbonTubeBuilder`, each building a :class:`~molpy.Frame`;
``mp.Atomistic.from_frame`` makes it a graph).

molpy adds:

* crystals — :class:`Lattice` / :class:`Site` / :class:`SpaceGroup`
  (a :class:`~molpy.Box` unit cell plus a basis; ``Lattice.from_cif`` expands
  a :func:`molpy.io.read_cif` frame by its symmetry operators;
  ``Lattice.build`` tiles it);
* polymers — sequence generators, chain-length distributions and
  :class:`SystemPlanner`; :class:`AmberPolymerBuilder` for GAFF chains
  through AmberTools;
* virtual sites — :class:`VirtualSiteBuilder` with :class:`DrudeBuilder`
  (CL&Pol, parameters from ``mp.ff.params.clpol_polarizability``) and
  :class:`Tip4pBuilder`;
* :class:`PackingTemplate` — a built molecule's frame plus its hydrogen
  indices, for a packer's per-atom settings.

Every name has this one path; the modules behind it are implementation
modules, outside ``__all__``.
"""

from molrs.builder import *  # noqa: F403
from molrs.builder import __all__ as _native

from molpy.builder.crystal import Lattice, Site
from molpy.builder.packing import PackingTemplate
from molpy.builder.polymer import (
    AlternatingSequenceGenerator,
    AmberBuildResult,
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
    BlockSequenceGenerator,
    Chain,
    DpDistribution,
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
from molpy.builder.symmetry import SpaceGroup
from molpy.builder.virtualsite import DrudeBuilder, Tip4pBuilder, VirtualSiteBuilder

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
    "DpDistribution",
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
