"""Force fields — :mod:`molrs.ff`, mirrored submodule by submodule.

``mp.ff`` has exactly the submodules ``molrs.ff`` has, and every native name
in them is the molrs object (``mp.ff.forcefield.ForceField is
molrs.ff.forcefield.ForceField``):

* :mod:`~molpy.ff.forcefield` — ``ForceField``, its ``Style`` / ``Type``
  handles, and the force-field file readers and writers
* :mod:`~molpy.ff.potential` — ``PotentialCompiler``, ``Potentials``,
  ``kernel``, ``LJCut`` and the ``Potential`` protocol
* :mod:`~molpy.ff.typifier` — the ``Typifier`` base, its ``Match``, the
  built-in typifiers and ``assign_cmaps``; plus molpy's AmberTools typifiers
* :mod:`~molpy.ff.charge` — partial-charge models
* :mod:`~molpy.ff.ir` — the force-field IR as a protocol (register a style or
  a category from Python)
* :mod:`~molpy.ff.params` — the parameter tables molrs ships
* :mod:`~molpy.ff.scale_lj` — CL&Pol fragment scaling of LJ parameters
"""

from . import charge, forcefield, ir, params, potential, scale_lj, typifier

__all__ = [
    "charge",
    "forcefield",
    "ir",
    "params",
    "potential",
    "scale_lj",
    "typifier",
]
