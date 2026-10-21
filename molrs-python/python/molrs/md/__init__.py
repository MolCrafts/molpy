"""In-process MD. One driver, :class:`MD`, walks the LAMMPS phase list.

The pieces are mounted on it. A potential is not an integrator argument::

    from molrs import md
    from molrs.ff.potential import PairLjCut

    sim = md.MD()
    sim.set_integrator(md.VelocityVerlet(1.0))
    sim.set_neighbor(md.NeighborList(6.8, 0.5))
    sim.set_evaluator(PairLjCut(0.238, 3.405, 6.8))
    sim.run(frame, 1000)

``run`` writes the same frame: coordinates, velocities, forces, image flags,
``meta["energy"]`` and ``meta["kinetic"]``. ``dt`` lives on the integrator.
Mass and velocities live on the frame. ``FixLangevin`` needs ``kb`` because
the driver does not pick a unit system.
"""

from .._native import md as _md

FixLangevin = _md.FixLangevin
MD = _md.MD
MaxwellBoltzmann = _md.MaxwellBoltzmann
NeighborList = _md.NeighborList
Phase = _md.Phase
VelocityVerlet = _md.VelocityVerlet

__all__ = [
    "FixLangevin",
    "MD",
    "MaxwellBoltzmann",
    "NeighborList",
    "Phase",
    "VelocityVerlet",
]
