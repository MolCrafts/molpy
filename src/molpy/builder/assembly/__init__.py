"""Assembly: paste graphs, apply a reaction locally, repair types locally.

One kernel (:class:`GraphAssembler`) and one variation point
(:class:`Selector`). Crosslinking is the kernel plus a proximity selector;
:class:`PolymerBuilder` is the kernel plus a monomer library and CGSmiles.
"""

from ._assembler import GraphAssembler
from ._context import MatchContext
from ._library import MonomerLibrary
from ._polymer import PolymerBuilder
from ._proximity import (
    Candidate,
    ExhaustiveSelector,
    ExplicitPairSelector,
    ProximitySelector,
    SpacingSelector,
)
from ._random import RandomSelector
from ._selector import Binding, Selector
from ._topology import TopologySelector

__all__ = [
    "Binding",
    "Candidate",
    "ExhaustiveSelector",
    "ExplicitPairSelector",
    "GraphAssembler",
    "MatchContext",
    "MonomerLibrary",
    "PolymerBuilder",
    "ProximitySelector",
    "RandomSelector",
    "Selector",
    "SpacingSelector",
    "TopologySelector",
]
