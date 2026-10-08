"""Force fields — :mod:`molrs.ff`, mirrored submodule by submodule.

``mp.ff`` has exactly the submodules ``molrs.ff`` has, and every native name
in them is the molrs object (``mp.ff.forcefield.ForceField is
molrs.ff.forcefield.ForceField``):

* :mod:`~molpy.ff.forcefield` — ``ForceField`` and its ``Style`` / ``ForceFieldType``
  handles (the data model; its file formats are :mod:`molpy.io`'s)
* :mod:`~molpy.ff.potential` — ``Potentials``, ``WeightedTerms``,
  ``PairLjCut`` and the ``Potential`` protocol
* :mod:`~molpy.ff.compile` — ``PotentialCompiler`` and
  ``ExplicitTerms``: a force field bound to its kernels
* :mod:`~molpy.ff.typifier` — the ``Typifier`` base, its ``TypeAssignment``, the
  built-in typifiers and ``assign_cmaps``; plus molpy's AmberTools typifiers
* :mod:`~molpy.ff.charge` — partial-charge models
* :mod:`~molpy.ff.ir` — the force-field IR as vocabulary: categories, styles,
  parameters and their refusals
* :mod:`~molpy.ff.style_registry` — register a style or a category from
  Python, with nothing rebuilt
* :mod:`~molpy.ff.params` — the parameter tables molrs ships
* :mod:`~molpy.ff.clpol_scaling` — CL&Pol fragment scaling of LJ parameters
"""

from . import (
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
