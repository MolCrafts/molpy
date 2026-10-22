"""MolPy — composable molecular modeling in Python, a thin layer over molrs.

Users write ``import molpy as mp``. Each name has exactly one public module;
the root holds three kinds of name and nothing else:

* **the subsystems** — one molpy module per molrs subsystem, each holding the
  same names by identity (``mp.perceive.SmartsPattern is
  molrs.perceive.SmartsPattern``): :mod:`molpy.core` (:mod:`molrs.core`,
  plus molpy's selectors and trajectory splitters), :mod:`molpy.io` (every file reader and
  writer, with one submodule per format that owns classes), :mod:`molpy.ff`,
  :mod:`molpy.perceive`, :mod:`molpy.optimize`, :mod:`molpy.conformer`,
  :mod:`molpy.builder`, :mod:`molpy.compute`, :mod:`molpy.md`,
  :mod:`molpy.op` and :mod:`molpy.stream`; and molpy's own
  :mod:`molpy.engine`, :mod:`molpy.tool`, :mod:`molpy.config` and
  :mod:`molpy.resources`;
* **the core data classes a user handles directly**, promoted from
  :mod:`molpy.core` as the same objects (``mp.Frame is mp.core.Frame is
  molrs.core.Frame``): ``Frame``, ``Block``, ``Trajectory``, ``Box``,
  ``MolGraph``, ``Atomistic``, ``CoarseGrain``, the entity classes (``Atom``,
  ``Bond``, ``Angle``, ``Dihedral``, ``Improper``, ``Bead``, ``CgBond``,
  ``Port``, ``VirtualSite``, ``DrudeParticle``, ``MasslessSite``),
  ``Element`` and ``Topology``. No function, algorithm or unit preset is
  promoted: everything else is reached through its subsystem
  (``mp.core.Cuboid``, ``mp.optimize.Lbfgs``, ``mp.io.read_pdb``);
* the version metadata: ``__version__`` and ``release_date``, and the
  :mod:`molpy.version` module that holds them with
  :func:`~molpy.version.check_molrs_version`.

The modules behind molpy's own additions (``molpy.engine.lammps``, …) are
implementation modules, outside every ``__all__``, so each of their names has
one path (``mp.engine.LammpsEngine``). No module name starts with ``_``:
``__all__``, not a name prefix, says what is public. Subsystems load lazily on
first attribute access (PEP 562).
"""

# Import version first: version.py runs the molcrafts-molrs compatibility check
# on import, before any molrs-backed core import below, so a stale editable
# build or a mismatched pin surfaces immediately. ``molpy.version`` is the
# module; the version string is ``__version__``.
from molpy import version

__version__ = version.version
release_date = version.release_date

from importlib import import_module as _import_module
from types import ModuleType as _ModuleType
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:
    from molpy import (
        builder,
        compute,
        config,
        conformer,
        core,
        engine,
        ff,
        io,
        md,
        op,
        optimize,
        perceive,
        resources,
        stream,
        tool,
    )

# Submodules are loaded lazily (PEP 562) so that importing a single
# subpackage (e.g. ``molpy.io``) does not eagerly initialize the
# whole io/engine surface. ``molpy.io`` et al. still work as
# attribute accesses and ``import molpy.io`` works as usual.
_LAZY_SUBMODULES = frozenset(
    {
        "builder",
        "compute",
        "config",
        "conformer",
        "core",
        "resources",
        "engine",
        "ff",
        "io",
        "md",
        "op",
        "optimize",
        "perceive",
        "stream",
        "tool",
    }
)


def __getattr__(name: str) -> _ModuleType:
    if name in _LAZY_SUBMODULES:
        return _import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_SUBMODULES)


# The core data classes a user handles directly, promoted from molpy.core
# (``mp.Frame is mp.core.Frame``). Nothing else from the core is on the root.
from molpy.core import (
    Angle,
    Atom,
    Atomistic,
    Bead,
    Block,
    Bond,
    Box,
    CgBond,
    CoarseGrain,
    Dihedral,
    DrudeParticle,
    Element,
    Frame,
    Improper,
    MasslessSite,
    MolGraph,
    Port,
    Topology,
    Trajectory,
    VirtualSite,
)

__all__ = [
    "builder",
    "compute",
    "config",
    "conformer",
    "core",
    "resources",
    "engine",
    "ff",
    "io",
    "md",
    "op",
    "optimize",
    "perceive",
    "stream",
    "tool",
    # promoted core data classes
    "Angle",
    "Atom",
    "Atomistic",
    "Bead",
    "Block",
    "Bond",
    "Box",
    "CgBond",
    "CoarseGrain",
    "Dihedral",
    "DrudeParticle",
    "Element",
    "Frame",
    "MolGraph",
    "Improper",
    "MasslessSite",
    "Port",
    "Topology",
    "Trajectory",
    "VirtualSite",
    "release_date",
    "version",
]
