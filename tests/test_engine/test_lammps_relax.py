"""LAMMPS engine unit tests — no binary, no subprocess.

Round-trip MD against a real ``lmp`` binary is out of scope for the unit
suite. Script generation and argument validation are what we own; those
are covered here and in ``test_engine.py`` (with mocked ``subprocess``).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import numpy as np
import pytest

import molpy as mp
from molpy.config import load_config
from molpy.engine import LammpsEngine
from molpy.ff.forcefield import ForceField
from molpy.wrapper import ToolError

_R0 = 1.5  # harmonic bond equilibrium length (Å)


def _dimer_system(separation: float = 2.2) -> tuple[mp.Frame, ForceField]:
    """A neutral C-C dimer (harmonic bond k=300, r0=1.5) and its force field."""
    ff = ForceField("dimer")
    c = ff.def_style("atom", "full").def_type("C", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("C-C", c, c, k=300.0, r0=_R0)
    ff.def_style("pair", "lj/cut/coul/cut").def_type("C", c, epsilon=0.05, sigma=3.4)

    frame = mp.Frame(
        {
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
        }
    )
    frame.box = mp.Box.cube(30.0)
    return frame, ff


def test_init_autodetects_executable() -> None:
    """``LammpsEngine()`` resolves a binary name without requiring it on PATH."""
    eng = LammpsEngine(check_executable=False)
    assert eng.executable in {"lmp", "lmp_serial", "lmp_mpi"}


def test_minimize_requires_box() -> None:
    """A box-free frame is rejected before any subprocess is launched."""
    frame, ff = _dimer_system()
    frame.box = None
    with pytest.raises(ValueError, match="periodic box"):
        LammpsEngine(check_executable=False).minimize(frame, ff)


def test_md_rejects_unknown_ensemble() -> None:
    """An unsupported ensemble fails fast, before any subprocess."""
    frame, ff = _dimer_system()
    with pytest.raises(ValueError, match="ensemble must be one of"):
        LammpsEngine(check_executable=False).md(frame, ff, ensemble="npt")


def test_a_run_that_writes_no_structure_raises_with_the_log_tail(tmp_path) -> None:
    """LAMMPS exits 0 but writes no relaxed.data: the error carries the log."""
    frame, _ = _dimer_system()
    ff = ForceField("dimer", units="real")
    c = ff.def_style("atom", "full").def_type("C", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("C-C", c, c, k=300.0, r0=_R0)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
        "C", c, epsilon=0.05, sigma=3.4
    )

    def no_output(argv, *, cwd, **_):
        (cwd / "log.lammps").write_text(
            "minimize 1e-4 1e-6 10 100\nWARNING: Lost atoms\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, None, None)

    engine = LammpsEngine(
        config=load_config({"engine": {"lammps": {"executable": "lmp"}}}),
        check_executable=False,
    )
    with (
        patch("subprocess.run", side_effect=no_output),
        pytest.raises(ToolError, match="WARNING: Lost atoms") as raised,
    ):
        engine.minimize(frame, ff, workdir=tmp_path)
    assert raised.value.tool == "engine.lammps"
    assert raised.value.returncode == 0
    assert "relaxed.data" in raised.value.reason
