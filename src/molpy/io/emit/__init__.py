"""Multi-engine input emitters for MolPy.

Each emitter produces a **complete input set** for its target MD engine —
not just a structure file. Given an ``Atomistic`` + ``ForceField`` the
emitter writes the data file, the force-field file, and a starter run
script into ``out_dir`` and returns the list of generated file paths. OpenMM
inputs come from :meth:`molpy.engine.OpenMMEngine.generate_inputs`.

Built-in emitters live on the one registry, :data:`emitters`::

    emitters.names()                          # ["gromacs", "lammps"]
    emitters.emit("lammps", atomistic, ff, out_dir, prefix="w")
    emitters.register("mine", MyEmitter())    # add an engine

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molrs import Atomistic
from molrs.ff import ForceField


class Emitter:
    """Base class for engine input emitters."""

    name: str = "base"

    def emit(
        self,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        **opts: Any,
    ) -> list[Path]:
        raise NotImplementedError


class EmitterRegistry:
    """Named engine emitters — the one public way to emit an input set."""

    def __init__(self) -> None:
        self._emitters: dict[str, Emitter] = {}

    def register(self, name: str, emitter: Emitter) -> None:
        """Register ``emitter`` under ``name``, replacing any previous one."""
        self._emitters[name] = emitter

    def names(self) -> list[str]:
        """Registered emitter names, sorted."""
        return sorted(self._emitters)

    def emit(
        self,
        name: str,
        atomistic: Atomistic,
        ff: ForceField,
        out_dir: Path,
        *,
        prefix: str = "system",
        **opts: Any,
    ) -> list[Path]:
        if name not in self._emitters:
            raise KeyError(
                f"Unknown emitter {name!r}. Registered: {sorted(self._emitters)}"
            )
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._emitters[name].emit(atomistic, ff, out_dir, prefix=prefix, **opts)


#: Built-in emitters, registered on import.
emitters = EmitterRegistry()


# Register built-in emitters on import
from .gromacs import GromacsEmitter
from .lammps import LammpsEmitter

emitters.register("lammps", LammpsEmitter())
emitters.register("gromacs", GromacsEmitter())

__all__ = [
    "Emitter",
    "EmitterRegistry",
    "emitters",
    "LammpsEmitter",
    "GromacsEmitter",
]
