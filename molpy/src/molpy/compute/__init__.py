"""Trajectory and structure analyses — :mod:`molrs.compute`, mirrored by identity.

Every analysis is the native class (``mp.compute.Rdf is molrs.compute.Rdf``);
molpy adds no wrapper. A compute is a configured instrument: its constructor
takes configuration only (bins, cutoffs, ``max_lag``, windows, tolerances), and
``compute(...)`` takes the data as keyword arguments — arrays, ``dt``,
neighbour tables, partitions, upstream results. A compute never reads a
``Frame``: the caller reads the columns it needs through the Frame/Block API
and hands the arrays over. Analysis time is femtoseconds (LAMMPS real units).

Pair-based analyses take neighbour tables from ``molpy.core.NeighborList``::

    >>> nl = mp.core.NeighborList(cutoff)
    >>> nl.build(frame.coords, frame.box)
    >>> g = mp.compute.Rdf(n_bins, cutoff, 0.0).compute(pairs=[nl.neighbors()])

Every time correlation runs on one engine, ``Correlator``. The named surfaces
``Msd``, ``Vacf``, ``OnsagerCorrelation``, ``DipoleAcf``, ``CurrentAcf``,
``ReorientationAcf``, ``RotationalAutocorrelation`` and ``Survival`` are
configurations of it, and each returns a ``TimeCorrelation``. A fit is a
compute whose input is that curve::

    raw curve  →  fit (LinearFit, Plateau, DebyeFit, CumulativeTrapezoid)

    >>> from molpy.compute import Msd, LinearFit
    >>> msd = Msd(max_lag=500).compute(positions=unwrapped, dt=10.0)
    >>> fit = LinearFit(0.1, 0.5).compute(correlation=msd, channel=0)
"""

from molrs.compute import *  # noqa: F403
from molrs.compute import __all__ as _native

__all__ = [*_native]
