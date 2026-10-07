"""Force fields — :mod:`molrs.ff`, mirrored submodule by submodule.

``mp.ff`` has exactly the submodules ``molrs.ff`` has, and every native name
in them is the molrs object (``mp.ff.forcefield.ForceField is
molrs.ff.forcefield.ForceField``):

* :mod:`~molpy.ff.forcefield` — ``ForceField`` and its ``Style`` / ``ForceFieldType``
  handles (the data model; its file formats are :mod:`molpy.io`'s)
* :mod:`~molpy.ff.potential` — ``PotentialCompiler``, ``Potentials``,
  ``compile_explicit_terms``, ``PairLjCut`` and the ``Potential`` protocol
* :mod:`~molpy.ff.typifier` — the ``Typifier`` base, its ``TypeAssignment``, the
  built-in typifiers and ``assign_cmaps``; plus molpy's AmberTools typifiers
* :mod:`~molpy.ff.charge` — partial-charge models
* :mod:`~molpy.ff.ir` — the force-field IR as a protocol (register a style or
  a category from Python)
* :mod:`~molpy.ff.params` — the parameter tables molrs ships
* :mod:`~molpy.ff.clpol_scaling` — CL&Pol fragment scaling of LJ parameters
"""

from . import charge, clpol_scaling, forcefield, ir, params, potential, typifier

__all__ = [
    "charge",
    "clpol_scaling",
    "forcefield",
    "ir",
    "params",
    "potential",
    "typifier",
]
