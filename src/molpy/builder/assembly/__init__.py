"""Assembly: compile local products, execute one reaction batch, then finalize.

One kernel (:class:`GraphAssembler`) and one variation point
(:class:`Selector`). Crosslinking is the kernel plus a proximity selector;
:class:`PolymerBuilder` is the kernel plus a monomer library and CGSmiles.
Typing caches scalar per-atom patches only; topology/bonded finalization is an
explicit independent stage.
"""

from ._assembler import GraphAssembler
from ._context import MatchContext
from ._finalize import AssemblyFinalizer, Finalization
from ._library import MonomerLibrary
from ._placer import Placer, ResiduePlacer
from ._polymer import PolymerBuilder
from ._proximity import (
    Candidate,
    ExhaustiveSelector,
    ExplicitPairSelector,
    ProximitySelector,
    SpacingSelector,
)
from ._random import RandomSelector
from ._replicas import Replicas
from ._residue_graph import linear_cgsmiles, ring_cgsmiles, star_cgsmiles
from ._selector import Binding, Selector
from ._sites import SiteMap
from ._topology import TopologySelector

__all__ = [
    "Binding",
    "AssemblyFinalizer",
    "Candidate",
    "ExhaustiveSelector",
    "ExplicitPairSelector",
    "GraphAssembler",
    "Finalization",
    "MatchContext",
    "MonomerLibrary",
    "Placer",
    "PolymerBuilder",
    "ResiduePlacer",
    "ProximitySelector",
    "RandomSelector",
    "Replicas",
    "Selector",
    "SiteMap",
    "SpacingSelector",
    "TopologySelector",
    "linear_cgsmiles",
    "ring_cgsmiles",
    "star_cgsmiles",
]
