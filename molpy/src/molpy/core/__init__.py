"""The core data model — :mod:`molrs.core`, mirrored by identity, plus molpy's tools over it.

Every public name of :mod:`molrs.core` is here as the molrs object
(``mp.core.Frame is molrs.core.Frame``): the column store and the frame
(``Block``, ``Frame`` and its metadata, ``Trajectory`` and its observables),
space (``Box``, neighbour search, geometric regions such as ``Cuboid``,
``Sphere`` and ``HalfSpace``, triangle meshes, point paths), the molecular
graph (``MolGraph``, ``Atomistic``, ``CoarseGrain`` and their live node /
relation views), elements, the unit engine (``UnitRegistry``, ``UnitPreset``,
``Quantity``, …). The vocabularies ``keys``, ``schema`` and ``constants``
are molpy submodules mirroring molrs's by identity, so ``from molpy.core.keys
import ELEMENT`` works and ``mp.core.keys.ELEMENT is molrs.core.keys.ELEMENT``.

molpy adds, acting on those types:

* the selectors — :class:`MaskPredicate`, :class:`ColumnSelector` (rows
  whose column equals a value) and :class:`AtomIndexSelector` (rows by
  0-based position) — boolean row masks over a ``Block`` that compose with
  ``&`` / ``|`` / ``~`` and with the geometric regions;
* :class:`TrajectorySplitter` and its strategies (:class:`SplitStrategy`,
  :class:`FrameIntervalStrategy`, :class:`TimeIntervalStrategy`,
  :class:`CustomStrategy`) — cutting a native ``Trajectory`` into segments.

The data classes a user handles directly (``Frame``, ``Block``,
``Trajectory``, ``Box``, ``MolGraph``, ``Atomistic``, ``CoarseGrain``, the
entity classes, ``Element``, ``Topology``) are also promoted to the ``molpy``
root, as the same objects (``mp.Frame is mp.core.Frame``).
"""

from importlib import import_module as _import_module

from molrs.core import *  # noqa: F403
from molrs.core import __all__ as _native

# The star import bound molrs's vocabulary modules under these names. Loading
# molpy's mirror of each rebinds the attribute to it (``from molpy.core import
# keys`` would not: it returns the existing attribute without importing).
for _vocabulary in ("constants", "keys", "schema"):
    _import_module(f"{__name__}.{_vocabulary}")

from molpy.core.selector import AtomIndexSelector, ColumnSelector, MaskPredicate
from molpy.core.splitter import (
    CustomStrategy,
    FrameIntervalStrategy,
    SplitStrategy,
    TimeIntervalStrategy,
    TrajectorySplitter,
)

__all__ = [
    *_native,
    "AtomIndexSelector",
    "ColumnSelector",
    "CustomStrategy",
    "FrameIntervalStrategy",
    "MaskPredicate",
    "SplitStrategy",
    "TimeIntervalStrategy",
    "TrajectorySplitter",
]
