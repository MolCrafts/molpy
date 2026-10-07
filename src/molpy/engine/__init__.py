"""External simulation engines.

:class:`Engine` is the abstract base for running an external program
(command construction, working directory, launcher and environment). Each
concrete engine has **one** input writer, ``generate_inputs``, and ``run``:

* :class:`LAMMPSEngine` — ``generate_inputs(frame, ff, out)`` writes the data
  file, force-field settings, init and input script; ``minimize`` / ``md``
  relax a frame through the same deck.
* :class:`GROMACSEngine` — ``generate_inputs(frame, ff, out)`` writes the
  ``.gro``, ``.top`` and ``.mdp`` templates; ``run`` grompp's and mdrun's an
  ``.mdp``.
* :class:`OpenMMEngine` — ``generate_inputs(frame, ff, config, out)`` writes
  the PDB, the force-field XML and a Python simulation script.
* :class:`CP2KEngine` — runs a CP2K input.

Scripts are :class:`Script` objects (editable text with a path)::

    from molpy.engine import LAMMPSEngine
    engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
    paths = engine.generate_inputs(frame, ff, "./calc")
    result = engine.run(Script.from_path(paths["input"]), workdir="./calc")
"""

from ._base import Engine
from ._cp2k import CP2KEngine
from ._gromacs import GROMACSEngine
from ._lammps import LAMMPSEngine
from ._openmm import OpenMMEngine, OpenMMSimulationConfig
from ._script import Script, ScriptLanguage

__all__ = [
    "CP2KEngine",
    "Engine",
    "GROMACSEngine",
    "LAMMPSEngine",
    "OpenMMEngine",
    "OpenMMSimulationConfig",
    "Script",
    "ScriptLanguage",
]
