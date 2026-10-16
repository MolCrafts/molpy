"""In-process MD: one ``Potential`` concept, Rust integrators, the ``MdDriver``.

End to end (Ar-like LJ dimer)::

    import numpy as np
    from molrs import md
    from molrs.ff.potential import PairLjCut
    from molrs.core import Box, Cell, NeighborList, VerletSkin

    pos = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
    rc, skin = 7.5, 1.0
    # search cutoff = rc + skin (what the engine indexes);
    # force cutoff = rc (what the potential sees); skin is the rebuild buffer.
    nl = VerletSkin(NeighborList(rc + skin), rc, pos, Box(Cell.lengths([20.0, 20.0, 20.0])), skin=skin)
    eps = 0.238  # caller units; MD does not convert
    vv = md.VelocityVerlet(1.0, potential=PairLjCut(eps, 3.405, rc),
                           neighbors=nl, mass=np.full(2, 39.948))
    state = vv.initial(pos, np.zeros_like(pos))
    state = vv.advance_n(state, 100)

The integrator owns the neighbour loop: it runs the skin's rebuild policy and
feeds fresh pairs to the nonbond potential; Python never does pair
bookkeeping.

Units contract — the engine is **unit-agnostic**. Take constants from
:class:`molrs.core.UnitPreset`::

    kb = molrs.core.UnitPreset("real").boltzmann()
    md.MaxwellBoltzmann(kb * 300.0, seed=0)
    md.MdDriver().run(frame, n, dt=dt, kb=kb, thermo=100)

MD defines no potential: it integrates a :class:`molrs.ff.potential.PairLjCut`,
a ``Potentials`` collection (e.g. from :class:`molrs.ff.compile.ExplicitTerms`), or
any object with ``evaluate(pos) -> (energy, forces)``. External forces (the
NN/Torch seam) subclass :class:`molrs.ff.potential.Potential`::

    class Spring(Potential):
        def evaluate(self, pos):
            return 0.05 * float((pos * pos).sum()), -0.1 * pos

ForceField + Frame runs go through the :class:`MdDriver` driver::

    md.MdDriver().set_forcefield(ff).set_neighbors(cutoff=rc, skin=2.0).run(
        frame, 1000, dt=1.0, kb=molrs.core.UnitPreset("real").boltzmann()
    )

Precision: ``MdDriver(dtype=np.float64)`` is the only entry. ``np.float32`` / mixed
raise; those loops belong in the Rust integrators.
"""

from .._native import md as _md

Langevin = _md.Langevin
MdState = _md.MdState
MaxwellBoltzmann = _md.MaxwellBoltzmann
VelocityVerlet = _md.VelocityVerlet

from ._driver import MdDriver

MdDriver.__module__ = __name__

__all__ = [
    "MdDriver",
    "Langevin",
    "MdState",
    "MaxwellBoltzmann",
    "VelocityVerlet",
]
