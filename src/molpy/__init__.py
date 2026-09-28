"""MolPy — Composable molecular modeling in Python.

Core data structures (``Atom``, ``ForceField``, ``Frame``, …) are imported
eagerly and exposed at the package root — users write ``import molpy as mp``
then ``mp.Frame`` (never ``molpy.core.Frame`` or the engine's ``Frame``). Heavier
subpackages (``io``, ``engine``, ``parser``, …) load lazily on first attribute
access (PEP 562).
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
        optimize,
        parser,
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
        "optimize",
        "parser",
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
# core/ — every public core symbol is a top-level export (not molpy.core.Foo)
# =============================================================================

from .core.atomistic import (
    Angle,
    Atom,
    Atomistic,
    Bond,
    Dihedral,
    DrudeParticle,
    Improper,
    MasslessSite,
    VirtualSite,
)
from .core.box import Box
from .core.cg import Bead, CGBond, CoarseGrain
from .core.config import Config
from .core.entity import (
    Entities,
    Entity,
    GraphViews,
    Link,
    NodeRef,
    Refs,
    RelationRef,
)
from .core import fields
from .core.forcefield import (
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondStyle,
    BondType,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairStyle,
    PairType,
    Parameters,
    PotentialCompiler,
    Style,
    Type,
)
from .core.ops import (
    FragmentScaling,
)
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

# Conformer: molpy subclass with Atomistic marshalling (not bare molrs.conformer.Conformer)
from .conformer import Conformer

# =============================================================================
# molrs facade — handwritten identity re-exports (users never import molrs)
# =============================================================================
# Only symbols not already bound above. ``molpy.X is molrs.X`` for each name
# here. Packages that collide with molpy (io / compute / typifier) are omitted.
# Subclasses with real molpy additions (Box, Trajectory, Conformer, UnitSystem)
# come from the molpy modules above, not from this block. Atomistic,
# CoarseGrain and Perceive are native identities. The base Region is molpy's
# own predicate model; its concrete regions (BoxRegion, SphereRegion and the
# combinators) subclass the native shapes.

# Import Frame/Block from the pure-Python layer path so static analysis
# (griffe/mkdocstrings) resolves ``molpy.Frame → molrs.frame.Frame`` without
# going through the top-level ``molrs.Frame`` re-export alias chain.
from molrs.frame import Block, Frame

# One record of trajectory-like data: an ordered sequence of frames sharing one
# identity (a single geometry is a length-1 collection; a scan or relaxation is
# longer). Downstream consumers (molnex, molhub) import this alias from here.
from collections.abc import Sequence as _Sequence
from typing import TypeAlias as _TypeAlias

# Explicit TypeAlias: with the module-level lazy ``__getattr__`` present, a
# bare implicit alias falls through to it in some checkers (ty resolved the
# name as ModuleType); the declared spelling pins it as a type alias.
FrameCollection: _TypeAlias = _Sequence[Frame]

from molrs import (
    BlockDtypeError,
    Cuboid,
    Element,
    ExtractedSubgraph,
    FRAME_SCHEMA_VERSION,
    FrameMeta,
    Graph,
    MetaDocument,
    MetaValue,
    NeighborList,
    NeighborQuery,
    Neighbors,
    Parallelepiped,
    Port,
    Quantity,
    Reaction,
    ScalarObservable,
    Sphere,
    Trace,
    Unit,
    UnitPreset,
    UnitRegistry,
    UnitsError,
    VectorObservable,
    VerletSkin,
    keys,
    schema,
    signal,
)
from molrs.builder import Assembler, AxisOrienter, GrowthPlacer, SitePlacer
from molrs.compute.density import (
    SpatialDistribution,
    SpatialDistributionResult,
)
from molrs.compute.distribution import (
    AngleDistribution,
    CombinedDistribution,
    CombinedDistributionResult,
    DihedralDistribution,
    DistanceDistribution,
    DistributionResult,
)
from molrs.compute.dynamics import (
    VanHove,
    VanHoveResult,
)
from molrs.compute.fitting import (
    CumulativeTrapezoid,
    LinearFit,
    Plateau,
)
from molrs.compute.hbond import (
    HBondCriterion,
    HBonds,
    HBondsResult,
)
from molrs.compute.order import (
    LegendreReorientation,
    LegendreReorientationResult,
)
from molrs.compute.spectroscopy import (
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
)
from molrs.compute.transport import (
    DebyeFit,
    DebyeRelaxation,
    EinsteinConductivity,
    EinsteinDiffusion,
    GreenKuboConductivity,
    GreenKuboDiffusion,
    VACF,
)
from molrs.compute.voronoi import (
    DensityGrid,
    MolecularMoments,
    RadicalVoronoi,
    VoronoiCells,
    VoronoiIntegration,
)
from molrs.conformer import (
    ConformerReport,
    ConformerStageReport,
)
from molrs.ff import (
    AtdTypifier,
    BccModel,
    GasteigerModel,
    MMFF94STypifier,
    MMFF94Typifier,
    MullikenModel,
    OPLSAATypifier,
    Potentials,
    Typifier,
)

# I/O is **only** on ``molpy.io`` (``mp.io.read_*`` / ``write_*``). Never re-export
# molrs.io / molrs.ff force-field file APIs / raw traj readers on the package root.
from molrs.io import (  # parser types and their error, not file I/O entries
    CGSmilesIR,
    SmilesError,
    SmilesIR,
)
from molrs.optimize import (
    LBFGS,
    OptReport,
)
from molrs.perceive import (
    Coarsener,
    Perceive,
    RingInfo,
    SmartsMatch,
    SmartsPattern,
    SubgraphMatcher,
)

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
    "optimize",
    "parser",
    "typifier",
    # Version
    "version",
    "release_date",
    # --- core/ (top-level) ---
    "Angle",
    "Atom",
    "Atomistic",
    "Bond",
    "Dihedral",
    "DrudeParticle",
    "Improper",
    "MasslessSite",
    "VirtualSite",
    "Box",
    "Bead",
    "CGBond",
    "CoarseGrain",
    "Config",
    "Entities",
    "Entity",
    "GraphViews",
    "Link",
    "NodeRef",
    "Refs",
    "RelationRef",
    "fields",
    "AngleStyle",
    "AngleType",
    "AtomStyle",
    "AtomType",
    "BondStyle",
    "BondType",
    "DihedralStyle",
    "DihedralType",
    "ForceField",
    "ImproperStyle",
    "ImproperType",
    "PairStyle",
    "PairType",
    "Parameters",
    "Style",
    "Type",
    "FragmentScaling",
    "AndRegion",
    "BoxRegion",
    "Cube",
    "NotRegion",
    "OrRegion",
    "Region",
    "SphereRegion",
    "Script",
    "ScriptLanguage",
    "AtomIndexSelector",
    "AtomTypeSelector",
    "CoordinateRangeSelector",
    "DistanceSelector",
    "ElementSelector",
    "MaskPredicate",
    "CustomStrategy",
    "FrameIntervalStrategy",
    "SplitStrategy",
    "TimeIntervalStrategy",
    "Trajectory",
    "TrajectorySplitter",
    "UnitSystem",
    "Conformer",
    # --- molrs identity re-exports ---
    "BlockDtypeError",
    "UnitsError",
    "NeighborList",
    "NeighborQuery",
    "Neighbors",
    "VerletSkin",
    "Block",
    "FRAME_SCHEMA_VERSION",
    "Frame",
    "FrameCollection",
    "FrameMeta",
    "MetaDocument",
    "MetaValue",
    "Quantity",
    "Unit",
    "UnitPreset",
    "UnitRegistry",
    "ScalarObservable",
    "VectorObservable",
    "SmilesIR",
    "CGSmilesIR",
    "SmilesError",
    "Port",
    "Cuboid",
    "Parallelepiped",
    "Sphere",
    "Element",
    "ExtractedSubgraph",
    "Graph",
    "Perceive",
    "Coarsener",
    "Trace",
    "Assembler",
    "AxisOrienter",
    "SitePlacer",
    "GrowthPlacer",
    "Reaction",
    "RingInfo",
    "SmartsMatch",
    "SmartsPattern",
    "SubgraphMatcher",
    "keys",
    "schema",
    "ConformerReport",
    "ConformerStageReport",
    "AtdTypifier",
    "BccModel",
    "GasteigerModel",
    "MMFF94STypifier",
    "MMFF94Typifier",
    "MullikenModel",
    "OPLSAATypifier",
    "Typifier",
    "LBFGS",
    "OptReport",
    "Potentials",
    "PotentialCompiler",
    "AngleDistribution",
    "CombinedDistribution",
    "CombinedDistributionResult",
    "DebyeFit",
    "DebyeRelaxation",
    "DensityGrid",
    "DihedralDistribution",
    "DistanceDistribution",
    "DistributionResult",
    "EinsteinConductivity",
    "EinsteinDiffusion",
    "EinsteinHelfandSpectrum",
    "GreenKuboConductivity",
    "GreenKuboDiffusion",
    "GreenKuboSpectrum",
    "HBondCriterion",
    "HBonds",
    "HBondsResult",
    "IRSpectrum",
    "LegendreReorientation",
    "LegendreReorientationResult",
    "LinearFit",
    "MolecularMoments",
    "Plateau",
    "PowerSpectrum",
    "RadicalVoronoi",
    "RamanSpectrum",
    "ResonanceRamanSpectrum",
    "RoaSpectrum",
    "CumulativeTrapezoid",
    "SpatialDistribution",
    "SpatialDistributionResult",
    "VACF",
    "VanHove",
    "VanHoveResult",
    "VcdSpectrum",
    "VoronoiCells",
    "VoronoiIntegration",
    "signal",
]
