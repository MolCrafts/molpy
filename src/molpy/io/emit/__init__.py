"""Multi-engine input emitters for MolPy.

Each emitter produces a **complete input set** for its target MD engine —
not just a structure file. Given an ``Atomistic`` + ``ForceField`` the
emitter writes the data file, the force-field file, and a starter run
script into ``out_dir`` and returns the list of generated file paths.

Registered emitters (lookup by name)::

    EMITTERS["lammps"]   ->  LammpsEmitter
    EMITTERS["openmm"]   ->  OpenMMEmitter
    EMITTERS["gromacs"]  ->  GromacsEmitter
    EMITTERS["xml"]      ->  XMLEmitter

Use :func:`emit_all` to run every registered emitter for ``--emit all``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField


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


EMITTERS: dict[str, Emitter] = {}


def register(name: str, emitter: Emitter) -> None:
    EMITTERS[name] = emitter


def emit(
    name: str,
    atomistic: Atomistic,
    ff: ForceField,
    out_dir: Path,
    *,
    prefix: str = "system",
    **opts: Any,
) -> list[Path]:
    if name not in EMITTERS:
        raise KeyError(f"Unknown emitter {name!r}. Registered: {sorted(EMITTERS)}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return EMITTERS[name].emit(atomistic, ff, out_dir, prefix=prefix, **opts)


def emit_all(
    atomistic: Atomistic,
    ff: ForceField,
    out_dir: Path,
    *,
    prefix: str = "system",
    **opts: Any,
) -> dict[str, list[Path]]:
    """Run every registered emitter; returns ``{engine: [paths]}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        name: e.emit(atomistic, ff, out_dir, prefix=prefix, **opts)
        for name, e in EMITTERS.items()
    }


# Register built-in emitters on import
from .gromacs import GromacsEmitter  # noqa: E402
from .lammps import LammpsEmitter  # noqa: E402
from .openmm import OpenMMEmitter  # noqa: E402
from .xml import XMLEmitter  # noqa: E402

register("lammps", LammpsEmitter())
register("openmm", OpenMMEmitter())
register("gromacs", GromacsEmitter())
register("xml", XMLEmitter())

__all__ = [
    "Emitter",
    "EMITTERS",
    "register",
    "emit",
    "emit_all",
    "LammpsEmitter",
    "OpenMMEmitter",
    "GromacsEmitter",
    "XMLEmitter",
]
