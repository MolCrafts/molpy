"""Assembly-specific finalization compatibility surface."""

from __future__ import annotations

from dataclasses import dataclass, field

from molpy.builder._finalize import Finalization, StructureFinalizer


@dataclass(frozen=True)
class AssemblyFinalizer(StructureFinalizer):
    """Finalize an assembled molecular graph, including aromaticity."""

    perceive_aromaticity: bool = field(default=True, init=False, repr=False)
