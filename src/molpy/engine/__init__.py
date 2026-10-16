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

One periodicity policy holds for every engine: a frame with a periodic box
is a periodic system; a frame without one (``Atomistic.to_frame()`` carries
none) is a non-periodic, vacuum system. LAMMPS shrink-wraps it
(``boundary s s s``) and OpenMM computes it without images (``NoCutoff`` by
default). A method that needs a periodic cell — Ewald / PME / PPPM, OpenMM's
``CutoffPeriodic``, a barostat — is refused for it before any file is
written, and so is GROMACS, whose cutoff scheme simulates periodic systems
only. The nonbonded settings (cutoffs, electrostatics) are the force
field's own, stated on its pair styles; an explicit argument overrides them.

Scripts are :class:`Script` values (a name, its text and an optional path)::

    from molpy.engine import LammpsEngine, Script
    engine = LammpsEngine()
    paths = engine.generate_inputs(frame, ff, "./calc")
    result = engine.run(Script.from_path(paths["input"]), workdir="./calc")

Each engine's executable, environment, launcher, environment variables and
time limit are its ``engine.<name>`` settings in molpy's configuration
(:mod:`molpy.config`), and each logs to ``molpy.engine.<name>``
(:mod:`mollog`): ``lammps``, ``gromacs``, ``openmm``, ``cp2k``. Every
failure is a :class:`molpy.wrapper.ToolError`.
"""

from molpy.engine._cp2k import Cp2kEngine
from molpy.engine._engine import Engine
from molpy.engine._gromacs import GromacsEngine
from molpy.engine._lammps import LammpsEngine
from molpy.engine._openmm import OpenmmEngine, OpenmmSimulationConfig
from molpy.engine._script import Script

__all__ = [
    "Cp2kEngine",
    "Engine",
    "GromacsEngine",
    "LammpsEngine",
    "OpenmmEngine",
    "OpenmmSimulationConfig",
    "Script",
]
