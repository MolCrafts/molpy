"""Shared optional topology finalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


from molpy.core.atomistic import Atomistic
from molrs.perceive import Perceive


class Finalization(StrEnum):
    """How far a newly built atomistic graph should be finalized."""

    ATOMS = "atoms"
    TOPOLOGY = "topology"


@dataclass(frozen=True)
class StructureFinalizer:
    """Apply the common topology tail after structure construction.

    Builders should create atoms and bonds first, then delegate here exactly
    once. ``ATOMS`` deliberately removes any inherited partial angles and
    dihedrals; ``TOPOLOGY`` regenerates the complete graph topology.
    """

    stage: Finalization = Finalization.TOPOLOGY
    perceive_aromaticity: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", Finalization(self.stage))

    def apply(self, graph: Atomistic) -> Atomistic:
        """Finalize ``graph`` and return it."""
        if self.stage is Finalization.ATOMS:
            graph.remove_link(*graph.angles, *graph.dihedrals)
            return graph

        graph.generate_topology(
            gen_angle=True,
            gen_dihedral=True,
            clear_existing=True,
        )
        if self.perceive_aromaticity:
            graph = Perceive().find_aromaticity(graph)
        return graph
