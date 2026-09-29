"""MolPy — Composable molecular modeling in Python.

Users write ``import molpy as mp`` and reach every public name through ``mp``
(``mp.Frame``, ``mp.Atomistic``, ``mp.io.read_pdb``). Each name has exactly one
public path; there is no ``molpy.core.X`` spelling of a root name.

Two layers own the names, and molpy says which on purpose:

**Re-exported from the native core (molrs).** Identity re-exports —
``mp.Atomistic is molrs.Atomistic`` — listed one by one at the bottom of this
module: the molecular graphs (``Atomistic``, ``CoarseGrain`` and their views
``Atom``/``Bond``/…/``Bead``/``CGBond``, ``NodeRef``/``RelationRef``/``Refs``,
``Port``, ``Graph``, ``Topology``, ``Trace``), tabular data (``Frame``,
``Block``, ``FrameMeta``, …), the force-field model (``ForceField``, the
``Style``/``Type`` trees, ``PotentialCompiler``, ``Potentials``), chemistry
notation and perception (``SmilesIR``, ``CGSmilesIR``, ``SmartsPattern``,
``Perceive``, ``RingInfo``, ``Reaction``, ``Coarsener``, ``SubgraphMatcher``),
site-graph assembly (``Assembler``, ``GrowthPlacer``, ``SitePlacer``,
``AxisOrienter``), geometric shapes (``Cuboid``, ``Sphere``,
``Parallelepiped``, ``HalfSpace``), units (``Unit``, ``Quantity``, ``UnitRegistry``, …),
neighbour search, charge models, ``LBFGS``/``OptReport`` and the conformer
reports. The namespaces ``mp.md`` and ``mp.op`` are native modules too.

**molpy's own.** Defined in molpy (subclasses marked *sub*): ``Box`` (sub),
``Trajectory`` (sub) and the ``TrajectorySplitter`` strategies, ``Region``
with ``BoxRegion``/``SphereRegion``/``Cube``/``AndRegion``/``OrRegion``/
``NotRegion`` (sub), the selectors, ``UnitSystem`` (sub), ``Conformer`` (sub),
``Config``, ``Script``, ``fields`` and ``FrameCollection``; plus the
subpackages ``io``, ``builder``, ``compute``, ``typifier``, ``engine``,
``adapter`` and ``data`` (and ``molpy.wrapper``, imported explicitly), which mix molpy code with native names
exported there and nowhere else (``mp.compute.RDF``,
``mp.typifier.OPLSAATypifier``).

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
        io,
        md,
        op,
        typifier,
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
        "io",
        "md",
        "op",
        "typifier",
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

from .core import fields
from .core.box import Box
from .core.config import Config
from .core.region import (
    AndRegion,
    BoxRegion,
    Cube,
    NotRegion,
    OrRegion,
    Region,
    SphereRegion,
)
from .core.script import Script, ScriptLanguage
from .core.selector import (
    AtomIndexSelector,
    AtomTypeSelector,
    CoordinateRangeSelector,
    DistanceSelector,
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
from .core.unit import UnitSystem
from .conformer import Conformer

# =============================================================================
# Re-exported from the native core — identity, ``molpy.X is molrs.X``
# =============================================================================
# Each name below has this one public path. Names that belong to a molpy
# subpackage namespace (analyses on ``mp.compute``, typifiers on
# ``mp.typifier``, file I/O on ``mp.io``) are exported there, not here.

from molrs import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Block,
    BlockDtypeError,
    Bond,
    CGBond,
    CoarseGrain,
    Cuboid,
    Dihedral,
    DrudeParticle,
    Element,
    ExtractedSubgraph,
    Frame,
    FrameMeta,
    Graph,
    HalfSpace,
    Improper,
    MasslessSite,
    MetaDocument,
    MetaValue,
    NeighborList,
    NeighborQuery,
    Neighbors,
    NodeRef,
    Parallelepiped,
    Port,
    Quantity,
    Reaction,
    Refs,
    RelationRef,
    ScalarObservable,
    Sphere,
    Topology,
    Trace,
    Unit,
    UnitPreset,
    UnitRegistry,
    UnitsError,
    VectorObservable,
    VerletSkin,
    VirtualSite,
    keys,
    schema,
)
from molrs.builder import Assembler, AxisOrienter, GrowthPlacer, SitePlacer
from molrs.conformer import ConformerReport, ConformerStageReport
from molrs.ff import (
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BccModel,
    BondStyle,
    BondType,
    DihedralStyle,
    DihedralType,
    ForceField,
    FragmentScaling,
    GasteigerModel,
    ImproperStyle,
    ImproperType,
    MullikenModel,
    PairStyle,
    PairType,
    PotentialCompiler,
    Potentials,
    Style,
    Type,
)

# Parser types and their error, not file I/O entries (those are on ``mp.io``).
from molrs.io import CGSmilesIR, SmilesError, SmilesIR
from molrs.optimize import LBFGS, OptReport
from molrs.perceive import (
    Coarsener,
    Perceive,
    RingInfo,
    SmartsMatch,
    SmartsPattern,
    SubgraphMatcher,
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
    "io",
    "md",
    "op",
    "typifier",
    # Version
    "version",
    "release_date",
    # --- molpy's own ---
    "AndRegion",
    "AtomIndexSelector",
    "AtomTypeSelector",
    "Box",
    "BoxRegion",
    "Config",
    "Conformer",
    "CoordinateRangeSelector",
    "Cube",
    "CustomStrategy",
    "DistanceSelector",
    "ElementSelector",
    "FrameCollection",
    "FrameIntervalStrategy",
    "MaskPredicate",
    "NotRegion",
    "OrRegion",
    "Region",
    "Script",
    "ScriptLanguage",
    "SphereRegion",
    "SplitStrategy",
    "TimeIntervalStrategy",
    "Trajectory",
    "TrajectorySplitter",
    "UnitSystem",
    "fields",
    # --- re-exported from the native core ---
    "Angle",
    "AngleStyle",
    "AngleType",
    "Assembler",
    "Atom",
    "AtomStyle",
    "AtomType",
    "Atomistic",
    "AxisOrienter",
    "BccModel",
    "Bead",
    "Block",
    "BlockDtypeError",
    "Bond",
    "BondStyle",
    "BondType",
    "CGBond",
    "CGSmilesIR",
    "Coarsener",
    "CoarseGrain",
    "ConformerReport",
    "ConformerStageReport",
    "Cuboid",
    "Dihedral",
    "DihedralStyle",
    "DihedralType",
    "DrudeParticle",
    "Element",
    "ExtractedSubgraph",
    "ForceField",
    "FragmentScaling",
    "Frame",
    "FrameMeta",
    "GasteigerModel",
    "Graph",
    "GrowthPlacer",
    "HalfSpace",
    "Improper",
    "ImproperStyle",
    "ImproperType",
    "LBFGS",
    "MasslessSite",
    "MetaDocument",
    "MetaValue",
    "MullikenModel",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "NodeRef",
    "OptReport",
    "PairStyle",
    "PairType",
    "Parallelepiped",
    "Perceive",
    "Port",
    "PotentialCompiler",
    "Potentials",
    "Quantity",
    "Reaction",
    "Refs",
    "RelationRef",
    "RingInfo",
    "ScalarObservable",
    "SitePlacer",
    "SmartsMatch",
    "SmartsPattern",
    "SmilesError",
    "SmilesIR",
    "Sphere",
    "Style",
    "SubgraphMatcher",
    "Topology",
    "Trace",
    "Type",
    "Unit",
    "UnitPreset",
    "UnitRegistry",
    "UnitsError",
    "VectorObservable",
    "VerletSkin",
    "VirtualSite",
    "keys",
    "schema",
]
