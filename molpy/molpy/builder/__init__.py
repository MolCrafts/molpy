"""Structure builders — ``molrs::builder``.

Graphene sheets (:class:`GrapheneBuilder`) and single-wall carbon nanotubes
(:class:`CarbonTubeBuilder`), each building a fresh :class:`molpy.core.Frame`.

Site-graph assembly: ``Assembler(library, SitePlacer(),
AxisOrienter()).assemble(sites)`` places one template copy per site of a
:class:`molpy.core.CoarseGrain`, joins bonded sites through their ports
(any topology), and returns the world as the library's graph type (every
template is one type: ``Atomistic``, ``CoarseGrain`` or ``MolGraph``);
``Assembler(library, GrowthPlacer()).assemble(sites)`` grows a site graph
without positions (e.g. ``read_cgsmiles(...).to_coarsegrain()``).

Coarse-graining: :class:`Coarsener` maps disjoint node groups of a held
``CoarseGrain`` or ``Atomistic`` onto the sites of a new ``CoarseGrain``, each
at its group's centre of mass with an axis from the group's first member;
each site's :meth:`~molpy.core.CoarseGrain.bead_members` are its group.

Built on those, this module also holds:

* crystals — :class:`Lattice` / :class:`Site` / :class:`SpaceGroup`
  (a :class:`~molpy.Box` unit cell plus a basis; ``Lattice.from_cif`` expands
  a :func:`molpy.io.read_cif` frame by its symmetry operators;
  ``Lattice.build`` tiles it);
* polymers — sequence generators, chain-length distributions and
  :class:`SystemPlanner`; :class:`AmberPolymerBuilder` for GAFF chains
  through AmberTools;
* virtual sites — :class:`DrudeBuilder` (CL&Pol, parameters from
  ``mp.ff.params.clpol_polarizability``) and :class:`Tip4pBuilder`;
* :class:`PackingTemplate` — a built molecule's frame plus its hydrogen
  indices, for a packer's per-atom settings.

The modules behind these are implementation modules, outside ``__all__``.
"""

from molpy._core import (
    Assembler,
    AxisOrienter,
    CarbonTubeBuilder,
    Coarsener,
    DrudeBuilder,
    GrapheneBuilder,
    GrowthPlacer,
    Lattice,
    Site,
    SitePlacer,
    SpaceGroup,
    SymOp,
    Tip4pBuilder,
)

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

__all__ = [
    "Assembler",
    "AxisOrienter",
    "CarbonTubeBuilder",
    "Coarsener",
    "GrapheneBuilder",
    "GrowthPlacer",
    "SitePlacer",
    "Lattice",
    "Site",
    "SpaceGroup",
    "SymOp",
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
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    "PackingTemplate",
]
