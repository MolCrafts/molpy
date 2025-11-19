"""Lightweight molpy package initializer for refactored core tests.

Avoid eager imports to prevent legacy dependencies from loading during
unit tests that target the new core architecture.
"""

# Submodules - Import these AFTER core classes to avoid circular imports
from . import (
    data,
    io,
    potential,
    typifier,
)

# Core atomistic classes
from .core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from .core.box import Box
from .core import Wrapper

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
from .version import version
