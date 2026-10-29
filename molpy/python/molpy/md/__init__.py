"""In-process MD. One driver, :class:`MD`, walks the LAMMPS phase list.

The pieces are mounted on it. A potential is not an integrator argument::

    import molpy as mp
    from molpy import md
    from molpy.ff.potential import PairLjCut

    sim = md.MD("real")
    sim.set_integrator(md.VelocityVerlet(10.0))
    sim.set_neighbor(mp.core.NeighborList(7.3), skin=0.5)
    sim.set_evaluator(PairLjCut(0.238, 3.405, 6.8))
    sim.run(frame, 1000)

A run is in one unit system, the preset ``MD`` is constructed with. ``dt``
is in its time unit (fs in ``real``, ps in ``metal``), as in LAMMPS. Masses,
energies and forces are in its units too. The neighbour list is the one
:class:`molpy.core.NeighborList`: it knows only its cutoff, and the frame box
supplies the periodicity. ``run`` writes the same frame: coordinates,
velocities, forces, image flags, ``meta["energy"]`` and ``meta["kinetic"]``.
``dt`` lives on the integrator; mass and velocities live on the frame. A force
provider of its own, such as a neural potential, is a :class:`Calculator`
mounted with :meth:`MD.set_calculator`.
"""

from molpy import _core
from molpy.md.calculator import Calculator

FixLangevin = _core.md.FixLangevin

MD = _core.md.MD

MaxwellBoltzmann = _core.md.MaxwellBoltzmann

Phase = _core.md.Phase

VelocityVerlet = _core.md.VelocityVerlet

__all__ = [
    "Calculator",
    "FixLangevin",
    "MD",
    "MaxwellBoltzmann",
    "Phase",
    "VelocityVerlet",
]
