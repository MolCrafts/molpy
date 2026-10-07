"""The core data model — molrs's core subsystems, mirrored by identity, plus molpy's tools over them.

molrs's core is four subsystems (Rust ``molrs::core::{store, system, spatial,
units}``); this one module holds the public names of all four, each the molrs
object (``mp.core.Frame is molrs.store.Frame``):

* :mod:`molrs.store` — ``Block``, ``Frame`` and its metadata, ``Trajectory``
  and its observables, the column vocabularies ``keys`` and ``schema``;
* :mod:`molrs.system` — the molecular-graph hierarchy (``Graph``,
  ``Atomistic``, ``CoarseGrain``) and its live node / relation views;
* :mod:`molrs.spatial` — ``Box``, neighbour search, geometric regions
  (``Cuboid``, ``Sphere``, ``HalfSpace``, …), triangle meshes, point paths;
* :mod:`molrs.units` — ``UnitRegistry``, ``UnitPreset``, ``Quantity``, ….

molpy adds, acting on those types:

* the column-value selectors — :class:`MaskPredicate` and
  :class:`ElementSelector`, :class:`AtomTypeSelector`,
  :class:`AtomIndexSelector` — boolean row masks over a ``Block`` that compose
  with ``&`` / ``|`` / ``~`` and with the geometric regions;
* :class:`TrajectorySplitter` and its strategies (:class:`SplitStrategy`,
  :class:`FrameIntervalStrategy`, :class:`TimeIntervalStrategy`,
  :class:`CustomStrategy`) — cutting a native ``Trajectory`` into segments.

The data classes a user handles directly (``Frame``, ``Block``,
``Trajectory``, ``Box``, ``Atomistic``, ``CoarseGrain``, the entity classes,
``Element``, ``Topology``) are also promoted to the ``molpy`` root, as the
same objects (``mp.Frame is mp.core.Frame``).
"""

from molrs.spatial import *  # noqa: F403
from molrs.spatial import __all__ as _spatial
from molrs.store import *  # noqa: F403
from molrs.store import __all__ as _store
from molrs.system import *  # noqa: F403
from molrs.system import __all__ as _system
from molrs.units import *  # noqa: F403
from molrs.units import __all__ as _units

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
    *_store,
    *_system,
    *_spatial,
    *_units,
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
