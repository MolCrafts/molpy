"""Trajectory and structure analyses — :mod:`molrs.compute`, mirrored by identity.

Every analysis is the native class (``mp.compute.Rdf is molrs.compute.Rdf``);
molpy adds no wrapper. Configure a compute, run ``.compute(...)`` on frames or
pre-assembled arrays, read typed fields. Analysis time is femtoseconds (LAMMPS
real units). FFT correlation, windows and frequency grids are
:mod:`molpy.signal`.

Pair-based analyses (``Rdf``, ``LocalDensity``, ``Steinhardt``,
``BondOrientationalOrder``, ``PmftXy``, ``Cluster``, …) take neighbour tables from
``molpy.core.NeighborList``::

    >>> nl = mp.core.NeighborList(cutoff)
    >>> nl.build(frame.coords, frame.box)
    >>> g = mp.compute.Rdf(n_bins, cutoff).compute([frame], [nl.neighbors()])

Transport and dielectric quantities are composed explicitly::

    raw curve  →  Fit  →  optional SI scale in your script

Example::

    >>> from molpy.compute import OnsagerCorrelation, EinsteinConductivity, LinearFit
    >>> L = OnsagerCorrelation().compute(P_i, P_j, 10.0, 500)
    >>> raw = EinsteinConductivity().compute(M, dt=10.0, max_correlation_time=500)
    >>> fit = LinearFit(0.1, 0.5).fit(raw["lag_times"], raw["msd"])
"""

from molrs.compute import *  # noqa: F403
from molrs.compute import __all__ as __all__
