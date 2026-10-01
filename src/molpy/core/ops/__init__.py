"""Operations over core objects that no single type owns.

Module-level functions here are the project's narrow OOP exception: they take
core objects and return values, with no state and no natural owning type. They
are reachable as ``molpy.core.ops.X`` and deliberately **not** as ``molpy.X`` —
the top-level facade carries core types and the ``read_*`` / ``write_*`` family,
nothing else.
"""

from molrs.ff import (
    compute_k_ij,
    fragment_scaling_data,
    intramolecular_pairs,
)

from .scale_lj import scale_lj

__all__ = [
    "compute_k_ij",
    "fragment_scaling_data",
    "intramolecular_pairs",
    "scale_lj",
]
