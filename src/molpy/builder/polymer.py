"""
Flexible PolymerBuilder for molpy - Template-based polymer construction.

This module provides a modular and extensible builder system for constructing
polymers using reusable monomer templates with context-aware anchor matching.
"""
from dataclasses import dataclass, field

from ..core.atomistic import Atomistic
from ..core.wrapper import Wrapper


@dataclass
class AnchorRule:
    """
    Context-aware anchor matching rule for polymer construction.

    Defines how an anchor atom should behave based on the context
    of neighboring monomers in the polymer chain.
    """

    init: int
    end: int
    deletes: list[int] = field(default_factory=list)


class Monomer(Wrapper):
    """
    Template for a monomer unit with anchor definitions.

    Inherits from Wrapper to enable composable functionality.
    Contains the structural information and anchor rules needed
    to construct and connect monomers in polymer chains.
    """

    anchors: list[AnchorRule] = field(default_factory=list)

    def __init__(
        self,
        struct: Atomistic,
        anchors: list[AnchorRule] = [],
    ):
        """Initialize Monomer with struct, anchors, and metadata."""
        super().__init__(struct)
        self.anchors = anchors

    
class PolymerBuilder:

    def __init__(self, monomers: dict[str, Monomer]):
        self.monomers = monomers
