"""MolPy — Composable molecular modeling in Python.

Core data structures (``Atom``, ``Frame``, ``ForceField``, …) are imported
eagerly. Heavier subpackages (``io``, ``engine``, ``parser``, …) are loaded
lazily on first attribute access via module ``__getattr__`` (PEP 562), so
``import molpy`` stays fast; ``molpy.io`` and ``import molpy.io`` work as
usual.
"""

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
        pack,
        parser,
        reacter,
        typifier,
    )

# Submodules are loaded lazily (PEP 562) so that importing a single
# subpackage (e.g. ``molpy.reacter``) does not eagerly initialize the
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
        "pack",
        "parser",
        "reacter",
        "typifier",
    }
)


def __getattr__(name: str) -> ModuleType:
    if name in _LAZY_SUBMODULES:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


# Core
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
from .core.entity import Entity, Link
from .core.forcefield import (
    AngleStyle,
    AngleType,
    AtomisticForcefield,
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
    Style,
    Type,
    TypeBucket,
)
from .core.frame import Block, Frame
from .core.script import Script, ScriptLanguage
from .core.trajectory import Trajectory
from .core.unit import UnitSystem
from .version import release_date, version

__all__ = [
    # Submodules
    "adapter",
    "builder",
    "compute",
    "data",
    "engine",
    "io",
    "pack",
    "parser",
    "reacter",
    "typifier",
    # Core atomistic
    "Angle",
    "Atom",
    "VirtualSite",
    "DrudeParticle",
    "MasslessSite",
    "Atomistic",
    "Bond",
    "Dihedral",
    "Improper",
    "Box",
    "Bead",
    "CGBond",
    "CoarseGrain",
    "Entity",
    "Link",
    # Core forcefield
    "AngleStyle",
    "AngleType",
    "AtomisticForcefield",
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
    "TypeBucket",
    # Core frame/topology/trajectory
    "Block",
    "Frame",
    "Script",
    "ScriptLanguage",
    "Trajectory",
    # Core units
    "UnitSystem",
    # Version
    "version",
    "release_date",
]
