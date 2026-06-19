"""LAMMPS structure-relaxation round-trip (``minimize`` / ``md``).

The relaxation tests exercise the full frame -> LAMMPS -> frame path and
therefore need a LAMMPS binary; they are marked ``external`` and skipped when
none is found.  The substrate is a neutral two-particle dimer with a single
harmonic bond (no angle/charge terms), so the energy minimum sits exactly at
the bond's ``r0`` and convergence is deterministic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import molrs
import numpy as np
import pytest

from molpy.core.forcefield import AtomStyle, BondStyle, ForceField, PairStyle
from molpy.engine import LAMMPS, LAMMPSEngine

_LMP = next((c for c in ("lmp", "lmp_serial", "lmp_mpi") if shutil.which(c)), None)

_R0 = 1.5  # harmonic bond equilibrium length (Å)


def _bond_length(frame: molrs.Frame) -> float:
    a = frame["atoms"]
    xyz = np.stack(
        [np.asarray(a.view("x")), np.asarray(a.view("y")), np.asarray(a.view("z"))],
        axis=1,
    )
    return float(np.linalg.norm(xyz[1] - xyz[0]))


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
            }
        }
    )
    frame.box = molrs.Box.cube(30.0)
    return frame, ff


def test_init_autodetects_executable() -> None:
    """``LAMMPSEngine()`` resolves a binary without an explicit name."""
    eng = LAMMPSEngine(check_executable=False)
    assert eng.executable in {"lmp", "lmp_serial", "lmp_mpi"}
    assert LAMMPS is LAMMPSEngine


def test_minimize_requires_box() -> None:
    """A box-free frame is rejected before any subprocess is launched."""
    frame, ff = _dimer_system()
    frame.box = None
    with pytest.raises(ValueError, match="periodic box"):
        LAMMPSEngine(check_executable=False).minimize(frame, ff)


@pytest.mark.external
@pytest.mark.skipif(_LMP is None, reason="no LAMMPS binary on PATH")
def test_minimize_restores_bond_length(tmp_path: Path) -> None:
    """A stretched bond relaxes back to the force field's r0."""
    frame, ff = _dimer_system(separation=2.2)
    assert _bond_length(frame) == pytest.approx(2.2)  # off-equilibrium input

    relaxed = LAMMPS(_LMP).minimize(frame, ff, workdir=tmp_path)

    assert isinstance(relaxed, molrs.Frame)
    assert _bond_length(relaxed) == pytest.approx(_R0, abs=1e-3)
    assert relaxed.box is not None  # box preserved
    assert _bond_length(frame) == pytest.approx(2.2)  # input not mutated


@pytest.mark.external
@pytest.mark.skipif(_LMP is None, reason="no LAMMPS binary on PATH")
def test_md_nve_limit_returns_finite_frame(tmp_path: Path) -> None:
    """A short nve/limit run returns a finite, complete frame."""
    frame, ff = _dimer_system(separation=_R0)
    relaxed = LAMMPS(_LMP).md(
        frame, ff, ensemble="nve/limit", steps=50, workdir=tmp_path
    )

    a = relaxed["atoms"]
    assert len(a.view("x")) == 2
    xyz = np.stack(
        [np.asarray(a.view("x")), np.asarray(a.view("y")), np.asarray(a.view("z"))],
        axis=1,
    )
    assert np.isfinite(xyz).all()


def test_md_rejects_unknown_ensemble() -> None:
    """An unsupported ensemble fails fast, before any subprocess."""
    frame, ff = _dimer_system()
    with pytest.raises(ValueError, match="ensemble must be one of"):
        LAMMPSEngine(check_executable=False).md(frame, ff, ensemble="npt")
