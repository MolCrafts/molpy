"""Assembly: execute one reaction batch, retype what it disturbed, then finalize.

One kernel (:class:`GraphAssembler`) and one variation point
(:class:`Selector`). Crosslinking is the kernel plus a proximity selector;
:class:`PolymerBuilder` is the kernel plus a monomer library and a residue topology.
Typing writes scalar per-atom data back only; topology/bonded finalization is an
explicit independent stage.
"""

from ._assembler import GraphAssembler
from ._context import MatchContext
from molpy.builder._finalize import Finalization

from ._finalize import AssemblyFinalizer
from ._library import Expansion, MonomerLibrary
from molrs import LineOrienter, Orienter, Placer, TangOrienter, Trace, TracePlacer
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
from ._residue_graph import (
    linear_topology,
    ring_topology,
    star_topology,
)
from ._residue_ir import ResidueBond, ResidueTopology, ResidueNode
from ._selector import Binding, Selector
from molrs import SiteMap
from ._topology import TopologySelector

__all__ = [
    "Binding",
    "AssemblyFinalizer",
    "Candidate",
    "ExhaustiveSelector",
    "Expansion",
    "ExplicitPairSelector",
    "GraphAssembler",
    "Finalization",
    "MatchContext",
    "MonomerLibrary",
    "LineOrienter",
    "Orienter",
    "Placer",
    "PolymerBuilder",
    "TangOrienter",
    "Trace",
    "TracePlacer",
    "ProximitySelector",
    "RandomSelector",
    "Replicas",
    "Selector",
    "SiteMap",
    "SpacingSelector",
    "TopologySelector",
    "ResidueBond",
    "ResidueTopology",
    "ResidueNode",
    "linear_topology",
    "ring_topology",
    "star_topology",
]
