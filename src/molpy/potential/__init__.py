"""molpy.potential — potentials and the force-field IR, re-exported by identity.

Everything here *is* its molrs object (``mp.potential.kernel is
molrs.ff.potential.kernel``); molpy keeps no parallel IR.

From :mod:`molrs.ff.potential`:

* :class:`Potential` — the protocol every Python force provider satisfies
  (``calc_energy_forces(pos) -> (energy, forces)``).
* :func:`kernel` — the kernel of any style the force-field IR prices,
  built-in or registered here, over explicit instances (atom indices and one
  parameter row per term, as stored: angle values in degrees). It returns a
  :class:`molpy.Potentials`, which ``push`` moves into a larger one::

      pots = mp.Potentials()
      pots.push(mp.potential.kernel("bond", "harmonic", [[0, 1]], k=300.0, r0=1.4))
      pots.push(mp.potential.kernel("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5))
      energy, forces = pots.calc_energy_forces(pos)

* :class:`LJCut` — the one-type ``lj/cut`` kernel the MD integrators feed
  from a neighbour list.

From :mod:`molrs.ff.ir`, the force-field IR as a protocol — a new style or
category from Python, nothing rebuilt:

* :class:`StyleSpec` — subclass it to declare and register a style (its
  ``category``, ``name``, ordered ``params`` with dimensions, and an
  ``expression`` or a ``kernel`` method);
* :class:`Param`, :func:`register_style`, :func:`register_category`,
  :func:`unregister`, :func:`styles`, :func:`categories`, :func:`evaluate`;
* :class:`IrError` — every refusal, a ``ValueError`` (its subclasses, one
  per refusal, are on :mod:`molrs.ff.ir`).

A bead-spring FENE bond (LAMMPS ``bond_style fene``) by its expression::

    class Fene(mp.potential.StyleSpec):
        category = "bond"
        name = "fene"
        params = {"k": "E/L^2", "r0": "L", "epsilon": "E", "sigma": "L"}
        expression = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
                      "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")
"""

from molrs.ff.ir import (
    IrError,
    Param,
    StyleSpec,
    categories,
    evaluate,
    register_category,
    register_style,
    styles,
    unregister,
)
from molrs.ff.potential import LJCut, Potential, kernel

__all__ = [
    "IrError",
    "LJCut",
    "Param",
    "Potential",
    "StyleSpec",
    "categories",
    "evaluate",
    "kernel",
    "register_category",
    "register_style",
    "styles",
    "unregister",
]
