"""CL&Pol fragment scaling of Lennard-Jones parameters — ``molrs::ff::clpol_scaling``.

:func:`scale_lj` returns a copy of a force field whose LJ well depths (and,
optionally, diameters) are scaled per fragment pair by the SAPT factor
:func:`compute_k_ij`; :class:`FragmentScaling` is one fragment's charge,
dipole and polarizability. The table molrs ships is
:func:`fragment_table`, which :func:`scale_lj` reads
when no ``fragment_table`` is given.
"""

from molpy._core import FragmentScaling, compute_k_ij, fragment_table, scale_lj

__all__ = [
    "FragmentScaling",
    "compute_k_ij",
    "fragment_table",
    "scale_lj",
]
