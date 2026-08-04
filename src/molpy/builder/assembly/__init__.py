"""Assembly: execute one reaction batch, retype what it disturbed, then finalize.

One kernel (:class:`GraphAssembler`) and one variation point
(:class:`Selector`). Crosslinking is the kernel plus a proximity selector;
:class:`PolymerBuilder` is the kernel plus a monomer library and a residue topology.
Typing writes scalar per-atom data back only; topology/bonded finalization is an
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
from ._residue_graph import (
    linear_cgsmiles,
    linear_topology,
    ring_cgsmiles,
    ring_topology,
    star_cgsmiles,
    star_topology,
)
from ._cgsmiles_ir import CGSmilesBondIR, CGSmilesGraphIR, CGSmilesNodeIR
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
    "CGSmilesBondIR",
    "CGSmilesGraphIR",
    "CGSmilesNodeIR",
    "linear_cgsmiles",
    "linear_topology",
    "ring_cgsmiles",
    "ring_topology",
    "star_cgsmiles",
    "star_topology",
]
