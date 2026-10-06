"""molpy.potential — the native ``molrs.ff.potential`` namespace re-exported by identity.

One kernel class per force-field style, built from explicit instances (atom
indices and one parameter row each, in the force field's convention: LAMMPS's,
angles in degrees) and moved into a :class:`molpy.Potentials` by ``push``, plus
the :class:`Potential` protocol every Python force provider satisfies::

    import molpy as mp
    from molpy.potential import AngleHarmonic, BondHarmonic

    pots = mp.Potentials()
    pots.push(BondHarmonic(atomi, atomj, k, r0))
    pots.push(AngleHarmonic(atomi, atomj, atomk, k, theta0_deg))
    energy, forces = pots.calc_energy_forces(pos)
"""

from molrs.ff.potential import (
    AngleHarmonic,
    BondHarmonic,
    DihedralPeriodic,
    ImproperCvff,
    ImproperPeriodic,
    LJCut,
    PairCoulCut,
    Potential,
)

__all__ = [
    "AngleHarmonic",
    "BondHarmonic",
    "DihedralPeriodic",
    "ImproperCvff",
    "ImproperPeriodic",
    "LJCut",
    "PairCoulCut",
    "Potential",
]
