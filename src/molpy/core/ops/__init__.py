"""Operations over core objects that no single type owns.

Module-level functions here are the project's narrow OOP exception: they take
core objects and return values, with no state and no natural owning type. They
are reachable as ``molpy.core.ops.X`` and deliberately **not** as ``molpy.X`` —
the top-level facade carries core types and the ``read_*`` / ``write_*`` family,
nothing else.
"""

from molrs.ff import extract_coords, fragment_scaling_data, intramolecular_pairs

from .scale_lj import (
    FragmentScaling,
    compute_k_ij,
    load_fragment_scaling_data,
    scale_lj,
)

__all__ = [
    "FragmentScaling",
    "compute_k_ij",
    "extract_coords",
    "fragment_scaling_data",
    "intramolecular_pairs",
    "load_fragment_scaling_data",
    "scale_lj",
]
