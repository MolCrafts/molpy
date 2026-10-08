"""External simulation engines.

:class:`Engine` is the abstract base for running an external program
(command construction, working directory, launcher and environment). Each
concrete engine has **one** input writer, ``generate_inputs``, and ``run``:

* :class:`LammpsEngine` — ``generate_inputs(frame, ff, out)`` writes the data
  file, force-field settings, init and input script; ``minimize`` / ``md``
  relax a frame through the same deck.
* :class:`GromacsEngine` — ``generate_inputs(frame, ff, out)`` writes the
  ``.gro``, ``.top`` and ``.mdp`` templates; ``run`` grompp's and mdrun's an
  ``.mdp``.
* :class:`OpenmmEngine` — ``generate_inputs(frame, ff, config, out)`` writes
  the PDB, the force-field XML and a Python simulation script.
* :class:`Cp2kEngine` — runs a CP2K input.

Scripts are :class:`Script` objects (editable text with a path)::

    from molpy.engine import LammpsEngine
    engine = LammpsEngine()
    paths = engine.generate_inputs(frame, ff, "./calc")
    result = engine.run(Script.from_path(paths["input"]), workdir="./calc")

Each engine's executable, environment, launcher, environment variables and
time limit are its ``engine.<name>`` settings in molpy's configuration
(:mod:`molpy.config`), and each logs to ``molpy.engine.<name>``
(:mod:`mollog`): ``lammps``, ``gromacs``, ``openmm``, ``cp2k``.
"""

from ._engine import Engine
from ._cp2k import Cp2kEngine
from ._gromacs import GromacsEngine
from ._lammps import LammpsEngine
from ._openmm import OpenmmEngine, OpenmmSimulationConfig
from ._script import Script, ScriptLanguage

__all__ = [
    "Cp2kEngine",
    "Engine",
    "GromacsEngine",
    "LammpsEngine",
    "OpenmmEngine",
    "OpenmmSimulationConfig",
    "Script",
    "ScriptLanguage",
]
