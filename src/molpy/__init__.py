"""MolPy — Composable molecular modeling in Python.

Core data structures (``Atom``, ``ForceField``, ``Frame``, …) are imported
eagerly and exposed at the package root — users write ``import molpy as mp``
then ``mp.Frame`` (never ``molpy.core.Frame`` or ``molrs.Frame``). Heavier
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
        optimize,
        pack,
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
        "optimize",
        "pack",
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
from .core.perceive import Perceive
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
    AngleClass2BondAngleStyle,
    AngleClass2BondBondStyle,
    AngleClass2Style,
    AngleHarmonicStyle,
    AngleStyle,
    AngleType,
    AtomisticForcefield,
    AtomStyle,
    AtomType,
    BondClass2Style,
    BondHarmonicStyle,
    BondMorseStyle,
    BondStyle,
    BondType,
    DihedralCharmmStyle,
    DihedralClass2Style,
    DihedralFourierStyle,
    DihedralMultiHarmonicStyle,
    DihedralOPLSStyle,
    DihedralPeriodicStyle,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperClass2Style,
    ImproperCvffStyle,
    ImproperHarmonicStyle,
    ImproperPeriodicStyle,
    ImproperStyle,
    ImproperType,
    PairBuckStyle,
    PairCoulLongStyle,
    PairCoulTTStyle,
    PairLjCutCoulCutStyle,
    PairLjCutCoulLongStyle,
    PairLJClass2Style,
    PairMorseStyle,
    PairStyle,
    PairTholeStyle,
    PairType,
    Parameters,
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
# Subclasses with real molpy additions (Box, Atomistic, CoarseGrain, Trajectory,
# Conformer, Region) come from core/ above, not from this block.

# Import Frame/Block from the pure-Python layer path so static analysis
# (griffe/mkdocstrings) resolves ``molpy.Frame → molrs.frame.Frame`` without
# going through the top-level ``molrs.Frame`` re-export alias chain.
from molrs.frame import Block, Frame  # noqa: E402

from molrs import (  # noqa: E402
    BlockDtypeError,
    Cuboid,
    Element,
    ExtractedSubgraph,
    FRAME_SCHEMA_VERSION,
    Graph,
    HollowSphere,
    MetaValue,
    MolRec,
    NeighborList,
    NeighborQuery,
    Neighbors,
    Observables,
    Parallelepiped,
    Quantity,
    Reaction,
    ScalarObservable,
    Sphere,
    Unit,
    UnitRegistry,
    UnitsError,
    VectorObservable,
    keys,
    schema,
    signal,
)
from molrs.compute.density import (  # noqa: E402
    SpatialDistribution,
    SpatialDistributionResult,
)
from molrs.compute.distribution import (  # noqa: E402
    AngleDistribution,
    CombinedDistribution,
    CombinedDistributionResult,
    DihedralDistribution,
    DistanceDistribution,
    DistributionResult,
)
from molrs.compute.dynamics import (  # noqa: E402
    VanHove,
    VanHoveResult,
)
from molrs.compute.fitting import (  # noqa: E402
    CumulativeTrapezoid,
    LinearFit,
    Plateau,
)
from molrs.compute.hbond import (  # noqa: E402
    HBondCriterion,
    HBonds,
    HBondsResult,
)
from molrs.compute.order import (  # noqa: E402
    LegendreReorientation,
    LegendreReorientationResult,
)
from molrs.compute.spectroscopy import (  # noqa: E402
    EinsteinHelfandSpectrum,
    GreenKuboSpectrum,
    IRSpectrum,
    PowerSpectrum,
    RamanSpectrum,
    ResonanceRamanSpectrum,
    RoaSpectrum,
    VcdSpectrum,
)
from molrs.compute.transport import (  # noqa: E402
    DebyeFit,
    DebyeRelaxation,
    EinsteinConductivity,
    EinsteinDiffusion,
    GreenKuboConductivity,
    GreenKuboDiffusion,
    VACF,
)
from molrs.compute.voronoi import (  # noqa: E402
    DensityGrid,
    MolecularMoments,
    RadicalVoronoi,
    VoronoiCells,
    VoronoiIntegration,
)
from molrs.conformer import (  # noqa: E402
    ConformerReport,
    ConformerStageReport,
)
from molrs.ff import (  # noqa: E402
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
from molrs.io import SmilesIR  # noqa: E402  # parser type, not a file I/O entry
from molrs.optimize import (  # noqa: E402
    LBFGS,
    OptReport,
)
from molrs.perceive import (  # noqa: E402
    RingInfo,
    SmartsMatch,
    SmartsPattern,
)

__all__ = [
    # Lazy subpackages
    "adapter",
    "builder",
    "compute",
    "data",
    "engine",
    "io",
    "optimize",
    "pack",
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
    "AngleClass2BondAngleStyle",
    "AngleClass2BondBondStyle",
    "AngleClass2Style",
    "AngleHarmonicStyle",
    "AngleStyle",
    "AngleType",
    "AtomisticForcefield",
    "AtomStyle",
    "AtomType",
    "BondClass2Style",
    "BondHarmonicStyle",
    "BondMorseStyle",
    "BondStyle",
    "BondType",
    "DihedralCharmmStyle",
    "DihedralClass2Style",
    "DihedralFourierStyle",
    "DihedralMultiHarmonicStyle",
    "DihedralOPLSStyle",
    "DihedralPeriodicStyle",
    "DihedralStyle",
    "DihedralType",
    "ForceField",
    "ImproperClass2Style",
    "ImproperCvffStyle",
    "ImproperHarmonicStyle",
    "ImproperPeriodicStyle",
    "ImproperStyle",
    "ImproperType",
    "PairBuckStyle",
    "PairCoulLongStyle",
    "PairCoulTTStyle",
    "PairLjCutCoulCutStyle",
    "PairLjCutCoulLongStyle",
    "PairLJClass2Style",
    "PairMorseStyle",
    "PairStyle",
    "PairTholeStyle",
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
    "Block",
    "FRAME_SCHEMA_VERSION",
    "Frame",
    "MetaValue",
    "Quantity",
    "Unit",
    "UnitRegistry",
    "ScalarObservable",
    "VectorObservable",
    "MolRec",
    "Observables",
    "SmilesIR",
    "Cuboid",
    "HollowSphere",
    "Parallelepiped",
    "Sphere",
    "Element",
    "ExtractedSubgraph",
    "Graph",
    "Perceive",
    "Reaction",
    "RingInfo",
    "SmartsMatch",
    "SmartsPattern",
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
