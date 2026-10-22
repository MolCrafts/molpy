"""LAMMPS engine relaxation tests — mocked ``subprocess``.

Script generation and argument validation are what we own; those are covered
here and in ``test_engine.py``. The one real minimisation runs only where an
``lmp`` binary is on ``PATH``.
"""

from __future__ import annotations

import shutil
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

import molpy as mp

_R0 = 1.5  # harmonic bond equilibrium length (Å)


def _dimer_system(
    separation: float = 2.2,
) -> tuple[mp.Frame, mp.ff.forcefield.ForceField]:
    """A neutral C-C dimer (harmonic bond k=300, r0=1.5) and its force field."""
    ff = mp.ff.forcefield.ForceField("dimer")
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
    frame.box = mp.Box(mp.core.Cell.lengths([30.0] * 3))
    return frame, ff


def _cut_dimer_system() -> tuple[mp.Frame, mp.ff.forcefield.ForceField]:
    """The dimer under a 10 Å ``lj/cut`` force field molrs can write."""
    frame, _ = _dimer_system()
    ff = mp.ff.forcefield.ForceField("dimer", units="real")
    c = ff.def_style("atom", "full").def_type("C", mass=12.011)
    ff.def_style("bond", "harmonic").def_type("C-C", c, c, k=300.0, r0=_R0)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0}).def_type(
        "C", c, epsilon=0.05, sigma=3.4
    )
    return frame, ff


def _lmp():
    """A config naming the ``lmp`` binary (not the auto-detected one)."""
    return mp.config.load_config({"engine": {"lammps": {"executable": "lmp"}}})


def test_init_autodetects_executable() -> None:
    """``LammpsEngine()`` resolves a binary name without requiring it on PATH."""
    eng = mp.engine.LammpsEngine(check_executable=False)
    assert eng.executable in {"lmp", "lmp_serial", "lmp_mpi"}


def test_minimize_relaxes_a_box_free_frame_shrink_wrapped(tmp_path) -> None:
    """A box-free frame is a non-periodic system: its deck is shrink-wrapped
    (``boundary s s s``), and the relaxed coordinates come back."""
    frame, ff = _cut_dimer_system()
    frame.box = None

    def lammps(argv, *, cwd, **_):
        shutil.copy(cwd / "system.data", cwd / "relaxed.data")
        return subprocess.CompletedProcess(argv, 0, None, None)

    engine = mp.engine.LammpsEngine(config=_lmp(), check_executable=False)
    with patch("subprocess.run", side_effect=lammps):
        relaxed = engine.minimize(frame, ff, workdir=tmp_path)
    init = (tmp_path / "system.in.init").read_text(encoding="utf-8").splitlines()
    assert "boundary s s s" in init
    assert relaxed.box is None
    np.testing.assert_allclose(relaxed["atoms"]["x"], frame["atoms"]["x"])


def test_a_long_range_force_field_needs_a_box(tmp_path) -> None:
    """PPPM needs a periodic cell: refused, by name, before any file."""
    frame, ff = _cut_dimer_system()
    frame.box = None
    ff.def_style("pair", "coul/long/pme", {"cutoff": 10.0})
    with pytest.raises(ValueError, match="LAMMPS.*long-range.*periodic box"):
        mp.engine.LammpsEngine(check_executable=False).minimize(
            frame, ff, workdir=tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_md_rejects_unknown_ensemble() -> None:
    """An unsupported ensemble fails fast, before any subprocess."""
    frame, ff = _dimer_system()
    with pytest.raises(ValueError, match="ensemble must be one of"):
        mp.engine.LammpsEngine(check_executable=False).md(frame, ff, ensemble="npt")


def test_a_run_that_writes_no_structure_raises_with_the_log_tail(tmp_path) -> None:
    """LAMMPS exits 0 but writes no relaxed.data: the error carries the log."""
    frame, ff = _cut_dimer_system()

    def no_output(argv, *, cwd, **_):
        (cwd / "log.lammps").write_text(
            "minimize 1e-4 1e-6 10 100\nWARNING: Lost atoms\n", encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, 0, None, None)

    engine = mp.engine.LammpsEngine(config=_lmp(), check_executable=False)
    with (
        patch("subprocess.run", side_effect=no_output),
        pytest.raises(mp.tool.ToolError, match="WARNING: Lost atoms") as raised,
    ):
        engine.minimize(frame, ff, workdir=tmp_path)
    assert raised.value.tool == "engine.lammps"
    assert raised.value.returncode == 0
    assert "relaxed.data" in raised.value.reason
