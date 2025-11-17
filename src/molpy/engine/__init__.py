"""
Engine module for molpy.

Provides abstract base classes and implementations for running external
computational chemistry programs like LAMMPS and CP2K.

The engine module integrates with the core Script class for input file
management. Scripts can be created from text, loaded from files, or loaded
from URLs.

Example:
    >>> from molpy.core.script import Script
    >>> from molpy.engine import LAMMPSEngine
    >>>
    >>> # Create input script
    >>> script = Script.from_text(
    ...     name="input",
    ...     text="units real\\natom_style full\\n",
    ...     language="other"
    ... )
    >>>
    >>> # Create engine and prepare
    >>> engine = LAMMPSEngine(executable="lmp")
    >>> engine.prepare(work_dir="./calc", scripts=script)
    >>>
    >>> # Run calculation
    >>> result = engine.run()
    >>> print(result.returncode)
    0
"""

from .base import Engine
from .cp2k import CP2KEngine
from .lammps import LAMMPSEngine

__all__ = ["CP2KEngine", "Engine", "LAMMPSEngine"]
