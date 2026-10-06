"""MolPy — composable molecular modeling in Python, a thin layer over molrs.

Users write ``import molpy as mp`` and reach every public name through ``mp``
(``mp.Frame``, ``mp.Atomistic``, ``mp.io.read_pdb``,
``mp.ff.forcefield.ForceField``). Each name has exactly one public path.

**What comes from molrs, and where.** Every native name is the molrs object
itself (``mp.Atomistic is molrs.system.Atomistic``); molpy keeps no parallel
IR, I/O, geometry, units or regions. molrs's subsystems map onto molpy in one
of two ways:

* *Mirrored as a subpackage* — the subsystems molpy also has a namespace for
  keep their molrs name and contents: :mod:`molpy.ff` (with ``forcefield``,
  ``potential``, ``typifier``, ``charge``, ``ir``, ``params``,
  ``scale_lj``), :mod:`molpy.io`, :mod:`molpy.compute`,
  :mod:`molpy.signal`, :mod:`molpy.md`, :mod:`molpy.op` and
  :mod:`molpy.builder`. molpy's own additions sit next to the native names
  there (the AmberTools typifiers in ``mp.ff.typifier``, ``read_smiles`` in
  ``mp.io``, crystals and polymers in ``mp.builder``).
* *Flattened onto this root* — the data model and the operations on it:
  ``molrs.store`` (``Frame``, ``Block``, ``keys``, ``schema``, …),
  ``molrs.system`` (``Atomistic``, ``CoarseGrain``, ``Graph`` and the live
  views), ``molrs.spatial`` (regions such as ``Cuboid`` and ``Sphere``,
  neighbour search), ``molrs.units``, ``molrs.perceive``,
  ``molrs.optimize`` and ``molrs.conformer``.

**molpy's own root names.** :class:`Box` (a subclass of the native box with a
free-box / diagonal constructor), :class:`Trajectory` (a subclass with a
topology, slicing and ``map``) and the :class:`TrajectorySplitter`
strategies, the column-value selectors (:class:`ElementSelector`, …) and
``FrameCollection``. molpy's own subpackages are :mod:`molpy.engine`,
:mod:`molpy.adapter`, :mod:`molpy.data` and :mod:`molpy.wrapper` (imported
explicitly).

Subpackages load lazily on first attribute access (PEP 562).
"""

# Import version first: version.py runs the molcrafts-molrs compatibility check
# on import, before any molrs-backed core import below, so a stale editable
# build or a mismatched pin surfaces immediately.
from .version import release_date, version

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        adapter,
        builder,
        compute,
        data,
        engine,
        ff,
        io,
        md,
        op,
        signal,
    )

# Submodules are loaded lazily (PEP 562) so that importing a single
# subpackage (e.g. ``molpy.io``) does not eagerly initialize the
# whole io/engine/adapter surface. ``molpy.io`` et al. still work as
# attribute accesses and ``import molpy.io`` works as usual.
_LAZY_SUBMODULES = frozenset(
    {
        "adapter",
        "builder",
        "compute",
        "data",
        "engine",
        "ff",
        "io",
        "md",
        "op",
        "signal",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


# =============================================================================
# molpy's own types
# =============================================================================

from .core.box import Box
from .core.selector import (
    AtomIndexSelector,
    AtomTypeSelector,
    ElementSelector,
    MaskPredicate,
)
from .core.trajectory import (
    CustomStrategy,
    FrameIntervalStrategy,
    SplitStrategy,
    TimeIntervalStrategy,
    Trajectory,
    TrajectorySplitter,
)

# =============================================================================
# molrs subsystems flattened onto the root — identity, ``mp.X is molrs.<sub>.X``
# =============================================================================
# ``Box`` and ``Trajectory`` are molpy's subclasses above, so the native ones
# are not re-exported under the same name.

from molrs.store import (
    Block,
    BlockDtypeError,
    Frame,
    FrameMeta,
    MetaDocument,
    MetaValue,
    ScalarObservable,
    VectorObservable,
    keys,
    schema,
)
from molrs.system import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Bond,
    CGBond,
    CoarseGrain,
    Dihedral,
    DrudeParticle,
    Element,
    ExtractedSubgraph,
    Graph,
    Improper,
    MasslessSite,
    NodeRef,
    Port,
    Refs,
    RelationBuckets,
    RelationRef,
    Topology,
    VirtualSite,
)
from molrs.spatial import (
    Cuboid,
    Cylinder,
    Ellipsoid,
    HalfSpace,
    NeighborList,
    NeighborQuery,
    Neighbors,
    Parallelepiped,
    Polyhedron,
    Region,
    Sphere,
    SphereUnion,
    Trace,
    TriMesh,
    VerletSkin,
)
from molrs.units import (
    AMBER_COULOMB,
    Quantity,
    Unit,
    UnitPreset,
    UnitRegistry,
    UnitsError,
)
from molrs.perceive import (
    Perceive,
    Reaction,
    RingInfo,
    SmartsMatch,
    SmartsPattern,
    SubgraphMatcher,
)
from molrs.optimize import (
    LBFGS,
    OptReport,
)
from molrs.conformer import (
    Conformer,
    ConformerReport,
    ConformerStageReport,
)

# One record of trajectory-like data (molpy's alias): an ordered sequence of
# frames sharing one identity (a single geometry is a length-1 collection; a
# scan or relaxation is longer). Downstream consumers (molnex, molhub) import
# this alias from here.
from collections.abc import Sequence as _Sequence
from typing import TypeAlias as _TypeAlias

# Explicit TypeAlias: with the module-level lazy ``__getattr__`` present, a
# bare implicit alias falls through to it in some checkers (ty resolved the
# name as ModuleType); the declared spelling pins it as a type alias.
FrameCollection: _TypeAlias = _Sequence[Frame]

__all__ = [
    # Lazy subpackages
    "adapter",
    "builder",
    "compute",
    "data",
    "engine",
    "ff",
    "io",
    "md",
    "op",
    "signal",
    # Version
    "version",
    "release_date",
    # --- molpy's own ---
    "AtomIndexSelector",
    "AtomTypeSelector",
    "Box",
    "CustomStrategy",
    "ElementSelector",
    "FrameCollection",
    "FrameIntervalStrategy",
    "MaskPredicate",
    "SplitStrategy",
    "TimeIntervalStrategy",
    "Trajectory",
    "TrajectorySplitter",
    # --- molrs.store ---
    "Block",
    "BlockDtypeError",
    "Frame",
    "FrameMeta",
    "MetaDocument",
    "MetaValue",
    "ScalarObservable",
    "VectorObservable",
    "keys",
    "schema",
    # --- molrs.system ---
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Bond",
    "CGBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "ExtractedSubgraph",
    "Graph",
    "Improper",
    "MasslessSite",
    "NodeRef",
    "Port",
    "Refs",
    "RelationBuckets",
    "RelationRef",
    "Topology",
    "VirtualSite",
    # --- molrs.spatial ---
    "Cuboid",
    "Cylinder",
    "Ellipsoid",
    "HalfSpace",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "Parallelepiped",
    "Polyhedron",
    "Region",
    "Sphere",
    "SphereUnion",
    "Trace",
    "TriMesh",
    "VerletSkin",
    # --- molrs.units ---
    "AMBER_COULOMB",
    "Quantity",
    "Unit",
    "UnitPreset",
    "UnitRegistry",
    "UnitsError",
    # --- molrs.perceive ---
    "Perceive",
    "Reaction",
    "RingInfo",
    "SmartsMatch",
    "SmartsPattern",
    "SubgraphMatcher",
    # --- molrs.optimize ---
    "LBFGS",
    "OptReport",
    # --- molrs.conformer ---
    "Conformer",
    "ConformerReport",
    "ConformerStageReport",
]
