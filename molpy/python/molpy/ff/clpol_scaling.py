"""CL&Pol fragment scaling of Lennard-Jones parameters — ``molrs::ff::clpol_scaling``.

:class:`ClpolScaling` holds the fragment table it reads (CL&Pol's own unless
one is given) and whether it scales σ; ``apply(ff, fragment_atoms)`` returns a
copy of a force field whose cross-fragment LJ well depths (and, optionally,
diameters) are scaled per fragment pair by the SAPT factor.
:class:`FragmentScaling` is one fragment's charge, dipole and polarizability,
and ``k_ij`` its SAPT factor with another fragment.
"""

from molpy._core import ClpolScaling, FragmentScaling

__all__ = [
    "ClpolScaling",
    "FragmentScaling",
]
