"""Force fields — ``molrs::ff``.

One submodule per Rust submodule, so the Python path and the Rust path are the
same words; this package itself holds nothing but them:

* :mod:`~molpy.ff.forcefield` — the :class:`~molpy.ff.forcefield.ForceField`
  data model with its ``Style`` / ``ForceFieldType`` handles (its files are
  :mod:`molpy.io`'s)
* :mod:`~molpy.ff.potential` — evaluable force terms: ``Potentials``,
  ``WeightedTerms``, ``PairLjCut``, and the ``Potential`` protocol
* :mod:`~molpy.ff.compile` — a force field bound to its kernels: the
  ``PotentialCompiler``, and ``ExplicitTerms`` for any style over
  explicit terms
* :mod:`~molpy.ff.typifier` — the subclassable ``Typifier`` base and its
  ``TypeAssignment``, the built-in typifiers (GAFF natively:
  ``AtdTypifier`` + ``GaffTypifier``), ``assign_cmaps``, and the AmberTools
  typifiers ``AntechamberTypifier`` / ``TleapTypifier``
* :mod:`~molpy.ff.charge` — partial-charge models (AM1-BCC / ABCG2,
  Mulliken, Gasteiger), and :class:`~molpy.ff.charge.Am1BccModel`, sqm's
  AM1 charges corrected natively
* :mod:`~molpy.ff.ir` — the force-field IR as vocabulary: categories,
  styles, parameters and their refusals
* :mod:`~molpy.ff.style_registry` — the style registry: register a new
  category or style (expression or Python kernel) with nothing rebuilt
* :mod:`~molpy.ff.params` — the parameter tables molrs ships (the CL&Pol
  polarizabilities and fragment scaling table)
* :mod:`~molpy.ff.clpol_scaling` — CL&Pol fragment scaling of Lennard-Jones
  parameters
"""

from molpy.ff import (
    charge,
    clpol_scaling,
    compile,
    forcefield,
    ir,
    params,
    potential,
    style_registry,
    typifier,
)

__all__ = [
    "charge",
    "clpol_scaling",
    "compile",
    "forcefield",
    "ir",
    "params",
    "potential",
    "style_registry",
    "typifier",
]
