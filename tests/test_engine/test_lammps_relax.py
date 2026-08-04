"""LAMMPS engine unit tests — no binary, no subprocess.

Round-trip MD against a real ``lmp`` binary is out of scope for the unit
suite. Script generation and argument validation are what we own; those
are covered here and in ``test_base.py`` (with mocked ``subprocess``).
"""

from __future__ import annotations

import molrs
import numpy as np
import pytest

import molpy.engine as molpy_engine
from molpy.core.forcefield import AtomStyle, BondStyle, ForceField, PairStyle
from molpy.engine import LAMMPSEngine

_R0 = 1.5  # harmonic bond equilibrium length (Å)


def _dimer_system(separation: float = 2.2) -> tuple[molrs.Frame, ForceField]:
    """A neutral C-C dimer (harmonic bond k=300, r0=1.5) and its force field."""
    ff = ForceField("dimer")
    carbon = ff.def_style(AtomStyle(name="full")).def_type("C", mass=12.011)
    ff.def_style(BondStyle(name="harmonic")).def_type(carbon, carbon, k=300.0, r0=_R0)
    ff.def_style(PairStyle(name="lj/cut/coul/cut")).def_type(
        carbon, carbon, epsilon=0.05, sigma=3.4
    )

    frame = molrs.Frame.from_dict(
        {
            "blocks": {
                "atoms": {
                    "x": np.array([0.0, separation]),
                    "y": np.zeros(2),
                    "z": np.zeros(2),
                    "type": ["C", "C"],
                    "charge": np.zeros(2),
                    "id": np.array([1, 2], dtype=np.int64),
                    "mol_id": np.array([1, 1], dtype=np.int64),
                },
                "bonds": {
                    "atomi": np.array([0], dtype=np.int64),
                    "atomj": np.array([1], dtype=np.int64),
                    "type": ["C-C"],
                },
            },
            "meta": {},
        }
    )
    frame.box = molrs.Box.cube(30.0)
    return frame, ff


def test_init_autodetects_executable() -> None:
    """``LAMMPSEngine()`` resolves a binary name without requiring it on PATH."""
    eng = LAMMPSEngine(check_executable=False)
    assert eng.executable in {"lmp", "lmp_serial", "lmp_mpi"}
    assert "LAMMPS" not in molpy_engine.__all__
    assert not hasattr(molpy_engine, "LAMMPS")


def test_minimize_requires_box() -> None:
    """A box-free frame is rejected before any subprocess is launched."""
    frame, ff = _dimer_system()
    frame.box = None
    with pytest.raises(ValueError, match="periodic box"):
        LAMMPSEngine(check_executable=False).minimize(frame, ff)


def test_md_rejects_unknown_ensemble() -> None:
    """An unsupported ensemble fails fast, before any subprocess."""
    frame, ff = _dimer_system()
    with pytest.raises(ValueError, match="ensemble must be one of"):
        LAMMPSEngine(check_executable=False).md(frame, ff, ensemble="npt")
