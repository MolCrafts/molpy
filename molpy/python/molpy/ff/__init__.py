"""Force fields — ``molrs::ff``.

The two classes that bind a force field to its kernels (``molrs::ff::compile``)
sit on this package itself:

* :class:`PotentialCompiler` — compiles a
  :class:`~molpy.ff.forcefield.ForceField` against a typed frame into
  :class:`~molpy.ff.potential.Potentials` (``compile``), or into
  :class:`~molpy.ff.potential.WeightedTerms` — each kernel with its
  special-bonds weights — for a neighbour-driven integrator
  (``compile_typed``).
* :class:`ExplicitTerms` — the terms of **any** style the force-field IR
  prices (a built-in, a style registered through
  :mod:`molpy.ff.style_registry` by expression or Python kernel, a style of a
  custom category) over explicit instances: atom indices and one parameter
  row per term, as stored (angle values in degrees). Its ``compile`` returns a
  :class:`~molpy.ff.potential.Potentials`::

      pots = Potentials()
      pots.push(ExplicitTerms("bond", "harmonic", [[0, 1], [1, 2]], k=300.0, r0=1.4).compile())
      pots.push(ExplicitTerms("angle", "harmonic", [[0, 1, 2]], k=50.0, theta0=109.5).compile())
      energy, forces = pots.evaluate(pos)

Everything else is one submodule per Rust submodule, so the Python path and
the Rust path are the same words:

* :mod:`~molpy.ff.forcefield` — the :class:`~molpy.ff.forcefield.ForceField`
  data model with its ``Style`` / ``ForceFieldType`` handles (its files are
  :mod:`molpy.io`'s)
* :mod:`~molpy.ff.potential` — evaluable force terms: ``Potentials``,
  ``WeightedTerms``, ``PairLjCut``, and the ``Potential`` protocol
* :mod:`~molpy.ff.typifier` — the subclassable ``Typifier`` base and its
  ``TypeAssignment``, the built-in typifiers (GAFF natively:
  ``AtdTypifier`` + ``GaffTypifier``), and the AmberTools
  typifiers ``AntechamberTypifier`` / ``TleapTypifier``
* :mod:`~molpy.ff.charge` — partial-charge models (AM1-BCC / ABCG2,
  Mulliken, Gasteiger), and :class:`~molpy.ff.charge.Am1BccModel`, sqm's
  AM1 charges corrected natively
* :mod:`~molpy.ff.ir` — the force-field IR as vocabulary: categories,
  styles, parameters and their refusals
* :mod:`~molpy.ff.style_registry` — the style registry: register a new
  category or style (expression or Python kernel) with nothing rebuilt
* :mod:`~molpy.ff.clpol_scaling` — CL&Pol fragment scaling of Lennard-Jones
  parameters
"""

from molpy._core import ExplicitTerms, PotentialCompiler
from molpy.ff import (
    charge,
    clpol_scaling,
    forcefield,
    ir,
    potential,
    style_registry,
    typifier,
)

__all__ = [
    "ExplicitTerms",
    "PotentialCompiler",
    "charge",
    "clpol_scaling",
    "forcefield",
    "ir",
    "potential",
    "style_registry",
    "typifier",
]
