"""External simulation engines.

:class:`Engine` is the abstract base for running an external program
(command construction, working directory, launcher and environment). Every
engine has ``run``. The LAMMPS, GROMACS and OpenMM engines each have **one**
input writer with one signature, ``generate_inputs(frame, forcefield,
output_dir, *, ...)``, returning the written paths by name; ``"input"`` is
always the runnable input ``run`` takes, and the other keys are the
engine's own:

* :class:`LammpsEngine` — ``"input"`` (the input script), ``"data"``,
  ``"settings"``, ``"init"``; ``minimize`` / ``md`` relax a frame through the
  same deck.
* :class:`GromacsEngine` — ``"input"`` (``em.mdp``), ``"gro"``, ``"top"``,
  ``"nvt"``; ``run`` grompp's and mdrun's an ``.mdp``, each stage from the
  structure the previous one wrote.
* :class:`OpenmmEngine` — ``"input"`` (the Python simulation script),
  ``"pdb"``, ``"forcefield"``; the simulation is the keyword-only
  ``simulation=`` (:class:`OpenmmSimulationConfig`).
* :class:`Cp2kEngine` — runs a CP2K input you write; it generates no inputs.

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
