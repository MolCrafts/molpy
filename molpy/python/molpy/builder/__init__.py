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
* polymers — the generative-BigSMILES model: a :class:`Fragment` is one
  ported template with a weight per port, a :class:`StochasticObject` grows
  repeat-unit and end-group fragments to a size drawn from a :class:`ChainSize` (a
  :class:`Distribution` counted in a measure), a :class:`Molecule` joins
  fixed fragments and objects end to end, and a :class:`System` samples
  molecule kinds to their :class:`Amount` of a total. ``sample(seed)`` gives
  the site graph an :class:`Assembler` builds over ``library()``;
  :func:`molpy.io.bigsmiles.read_bigsmiles` reads the same model from a
  string. A refusal is a :class:`PolymerError` (a ``ValueError``) whose
  ``kind`` tells an invalid model from a sample that got stuck. :class:`AmberPolymerBuilder` builds GAFF chains through
  AmberTools;
* virtual sites — :class:`DrudeBuilder` (CL&Pol; its shipped table is
  ``DrudeBuilder().polarizability``) and :class:`Tip4pBuilder`;
* :class:`PackingTemplate` — a built molecule's frame plus its hydrogen
  indices, for a packer's per-atom settings.

The modules behind these are implementation modules, outside ``__all__``.
"""

from molpy._core import (
    Amount,
    Assembler,
    AxisOrienter,
    CarbonTubeBuilder,
    ChainSize,
    Coarsener,
    Distribution,
    DrudeBuilder,
    Fragment,
    GrapheneBuilder,
    GrowthPlacer,
    JunctionReport,
    Lattice,
    Molecule,
    MoleculeRecord,
    PolymerError,
    Patch,
    Site,
    SitePlacer,
    SpaceGroup,
    StochasticObject,
    SymOp,
    System,
    SystemSample,
    Terminal,
    Tip4pBuilder,
    update_junctions,
)

from molpy.builder.packing import PackingTemplate
from molpy.builder.polymer import (
    AmberBuildResult,
    AmberCut,
    AmberPieces,
    AmberPolymerBuilder,
)

__all__ = [
    "Assembler",
    "AxisOrienter",
    "CarbonTubeBuilder",
    "Coarsener",
    "GrapheneBuilder",
    "GrowthPlacer",
    "JunctionReport",
    "SitePlacer",
    "Lattice",
    "Patch",
    "Site",
    "SpaceGroup",
    "SymOp",
    "Amount",
    "ChainSize",
    "Distribution",
    "Molecule",
    "MoleculeRecord",
    "PolymerError",
    "StochasticObject",
    "System",
    "SystemSample",
    "Fragment",
    "Terminal",
    "AmberBuildResult",
    "AmberCut",
    "AmberPieces",
    "AmberPolymerBuilder",
    "DrudeBuilder",
    "Tip4pBuilder",
    "update_junctions",
    "PackingTemplate",
]
