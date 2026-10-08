"""The core data model — :mod:`molrs.core`, mirrored by identity, plus molpy's tools over it.

Every public name of :mod:`molrs.core` is here as the molrs object
(``mp.core.Frame is molrs.core.Frame``): the column store and the frame
(``Block``, ``Frame`` and its metadata, ``Trajectory`` and its observables),
space (``Box``, neighbour search, geometric regions such as ``Cuboid``,
``Sphere`` and ``HalfSpace``, triangle meshes, point paths), the molecular
graph (``MolGraph``, ``Atomistic``, ``CoarseGrain`` and their live node /
relation views), elements, the unit engine (``UnitRegistry``, ``UnitPreset``,
``Quantity``, …) and the vocabularies ``keys``, ``schema`` and ``constants``
(``mp.core.keys is molrs.core.keys``).

molpy adds, acting on those types:

* the column-value selectors — :class:`MaskPredicate` and
  :class:`ElementSelector`, :class:`AtomTypeSelector`,
  :class:`AtomIndexSelector` — boolean row masks over a ``Block`` that compose
  with ``&`` / ``|`` / ``~`` and with the geometric regions;
* :class:`TrajectorySplitter` and its strategies (:class:`SplitStrategy`,
  :class:`FrameIntervalStrategy`, :class:`TimeIntervalStrategy`,
  :class:`CustomStrategy`) — cutting a native ``Trajectory`` into segments.

The data classes a user handles directly (``Frame``, ``Block``,
``Trajectory``, ``Box``, ``MolGraph``, ``Atomistic``, ``CoarseGrain``, the
entity classes, ``Element``, ``Topology``) are also promoted to the ``molpy``
root, as the same objects (``mp.Frame is mp.core.Frame``).
"""

from molrs.core import *  # noqa: F403
from molrs.core import __all__ as _native

from ._selector import (
    AtomIndexSelector,
    AtomTypeSelector,
    ElementSelector,
    MaskPredicate,
)
from ._splitter import (
    CustomStrategy,
    FrameIntervalStrategy,
    SplitStrategy,
    TimeIntervalStrategy,
    TrajectorySplitter,
)

__all__ = [
    *_native,
    "AtomIndexSelector",
    "AtomTypeSelector",
    "CustomStrategy",
    "ElementSelector",
    "FrameIntervalStrategy",
    "MaskPredicate",
    "SplitStrategy",
    "TimeIntervalStrategy",
    "TrajectorySplitter",
]
