"""
Engine module for MolPy.

Provides :class:`~molpy.engine.base.Engine`, an abstract base for running
external computational chemistry programs, together with concrete
implementations for LAMMPS, CP2K, and OpenMM.

Two usage modes are supported:

* **Generate only** — write input files without executing::

      paths = engine.generate_inputs(frame, ff, config, "./output")

* **Generate and run** — write files then launch the subprocess::

      result = engine.run(script, workdir="./calc")

MPI and job-scheduler launchers are supported via the ``launcher`` parameter::

    from molpy.engine import LAMMPSEngine
    engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])
    result = engine.run(script, workdir="./calc")
"""

from .base import Engine
from .cp2k import CP2KEngine
from .lammps import LAMMPSEngine
from .openmm import OpenMMEngine, OpenMMSimulationConfig

__all__ = [
    "CP2KEngine",
    "Engine",
    "LAMMPSEngine",
    "OpenMMEngine",
    "OpenMMSimulationConfig",
]
