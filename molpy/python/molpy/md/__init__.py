"""In-process MD. One driver, :class:`MD`, walks the LAMMPS phase list.

The pieces are mounted on it. A potential is not an integrator argument::

    import molpy as mp
    from molpy import md
    from molpy.ff.potential import PairLjCut

    sim = md.MD()
    sim.set_integrator(md.VelocityVerlet(1.0))
    sim.set_neighbor(mp.core.NeighborList(7.3), skin=0.5)
    sim.set_evaluator(PairLjCut(0.238, 3.405, 6.8))
    sim.run(frame, 1000)

The neighbour list is the one :class:`molpy.core.NeighborList`: it knows only
its cutoff, and the frame box supplies the periodicity. ``run`` writes the
same frame: coordinates, velocities, forces, image flags,
``meta["energy"]`` and ``meta["kinetic"]``. ``dt`` lives on the integrator.
Mass and velocities live on the frame. ``FixLangevin`` needs ``kb`` because
the driver does not pick a unit system.
"""

from molpy import _core

FixLangevin = _core.md.FixLangevin

MD = _core.md.MD

MaxwellBoltzmann = _core.md.MaxwellBoltzmann

Phase = _core.md.Phase

VelocityVerlet = _core.md.VelocityVerlet

__all__ = [
    "FixLangevin",
    "MD",
    "MaxwellBoltzmann",
    "Phase",
    "VelocityVerlet",
]
