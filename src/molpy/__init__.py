"""Lightweight molpy package initializer for refactored core tests.

Avoid eager imports to prevent legacy dependencies from loading during
unit tests that target the new core architecture.
"""

# Submodules - Import these AFTER core classes to avoid circular imports
from . import data, external, io, parser, potential, typifier
from .core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from .core.box import Box

# Core atomistic classes
from .core.entity import Entity, Link, Struct

# Core forcefield classes
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

# Core frame and box classes
from .core.frame import Block, Frame

# Core script classes
from .core.script import Script, ScriptLanguage

# Core topology class
from .core.topology import Topology
from .core.trajectory import Trajectory
from .potential import *
from .version import __version__, version
