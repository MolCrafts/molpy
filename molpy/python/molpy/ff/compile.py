"""A force field bound to its kernels — ``molrs::ff::compile``.

* :class:`PotentialCompiler` — compiles a
  :class:`~molpy.ff.forcefield.ForceField` against a typed frame into
  :class:`~molpy.ff.potential.Potentials` (``compile``), or into
  :class:`~molpy.ff.potential.WeightedTerms` — each kernel with its
  special-bonds weights — for a neighbour-driven integrator
  (``compile_typed``).
* :class:`ExplicitTerms` — the terms of **any** style the
  force-field IR prices (a built-in, a style registered through
  :mod:`molpy.ff.style_registry` by expression or Python kernel, a style of a
  custom category) over explicit instances: atom indices and one parameter
  row per term, as stored (angle values in degrees). Its ``compile`` returns a
  :class:`~molpy.ff.potential.Potentials`::

      pots = Potentials()
      pots.push(ExplicitTerms("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4).compile())
      pots.push(ExplicitTerms("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5).compile())
      energy, forces = pots.evaluate(pos)
"""

from molpy._core import ExplicitTerms, PotentialCompiler

__all__ = [
    "ExplicitTerms",
    "PotentialCompiler",
]
