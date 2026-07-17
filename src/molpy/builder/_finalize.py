"""Shared optional topology and bonded-parameter finalization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import molrs

from molpy.core.atomistic import Atomistic
from molpy.typifier.forcefield import ForceFieldParams


class Finalization(StrEnum):
    """How far a newly built atomistic graph should be finalized."""

    ATOMS = "atoms"
    TOPOLOGY = "topology"
    BONDED = "bonded"


@dataclass(frozen=True)
class StructureFinalizer:
    """Apply the common topology/bonded tail after structure construction.

    Builders should create atoms and bonds first, then delegate here exactly
    once.  ``ATOMS`` deliberately removes any inherited partial angles and
    dihedrals; ``TOPOLOGY`` regenerates the complete graph topology; and
    ``BONDED`` additionally resolves force-field types and parameters.
    """

    stage: Finalization = Finalization.TOPOLOGY
    bonded: ForceFieldParams | None = None
    perceive_aromaticity: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", Finalization(self.stage))
        if self.stage is Finalization.BONDED and self.bonded is None:
            raise TypeError("Finalization.BONDED requires bonded=ForceFieldParams(...)")
        if self.stage is not Finalization.BONDED and self.bonded is not None:
            raise TypeError("bonded= is only meaningful with Finalization.BONDED")

    def apply(self, graph: Atomistic) -> Atomistic:
        """Finalize ``graph`` and return it (or the parameterized copy)."""
        if self.stage is Finalization.ATOMS:
            graph.del_angle(*tuple(graph.angles))
            graph.del_dihedral(*tuple(graph.dihedrals))
            return graph

        graph.generate_topology(
            gen_angle=True,
            gen_dihedral=True,
            clear_existing=True,
        )
        if self.perceive_aromaticity:
            molrs.perceive_aromaticity(graph)
        if self.stage is Finalization.BONDED:
            assert self.bonded is not None
            return self.bonded.assign(graph)
        return graph
