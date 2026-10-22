"""Urey–Bradley across the FFI seam: ``angle charmm``.

LAMMPS ``angle_style charmm``: E = K(θ − θ0)² + K_ub(r13 − r_ub)², with the
four params ``k`` (energy/rad²), ``theta0`` (degrees), ``k_ub``
(energy/length²) and ``r_ub`` (length). The kernel, its forces and the LAMMPS
comparison are tested in Rust (``ff::potential::angle::charmm``); these tests
prove the Python surface: an ``AngleStyle`` named ``charmm`` takes the four
params, compiles to the LAMMPS energy, and survives the ``forcefield`` section,
a ``*.mrec`` store and the LAMMPS writer and reader.
"""

from __future__ import annotations

import math
from pathlib import Path

import molpy as mp
import numpy as np


def _lammps_include(
    ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame, **options: object
) -> str:
    """The LAMMPS force-field include ``ff`` writes for ``frame``."""
    buf = bytearray()
    mp.io.lammps.LammpsForcefieldWriter(frame, **options).write(buf, ff)  # type: ignore[arg-type]
    return buf.decode()


def _lammps_data_coeffs(ff: mp.ff.forcefield.ForceField, frame: mp.core.Frame) -> str:
    """The data file's ``* Coeffs`` sections ``ff`` writes for ``frame``."""
    buf = bytearray()
    mp.io.lammps.LammpsForcefieldWriter(frame).write_data_coeffs(buf, ff)
    return buf.decode()


K, THETA0, K_UB, R_UB = 33.43, 110.1, 22.53, 2.179
XYZ = np.array([[1.53, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.30, 1.05, 0.10]])


def _ub_ff() -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("charmm", units="real")
    atoms = ff.def_style("atom", "full")
    ct = atoms.def_type("CT", mass=12.011)
    ha = atoms.def_type("HA", mass=1.008)
    ff.def_style("angle", "charmm").def_type(
        "HA-CT-CT", ha, ct, ct, k=K, theta0=THETA0, k_ub=K_UB, r_ub=R_UB
    )
    return ff


def _frame() -> mp.core.Frame:
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, XYZ[:, d].copy())
    atoms.insert("type", ["HA", "CT", "CT"])
    angles = mp.core.Block()
    for key, atom in (("atomi", 0), ("atomj", 1), ("atomk", 2)):
        angles.insert(key, np.array([atom], dtype=np.uint32))
    angles.insert("type", ["HA-CT-CT"])
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame["angles"] = angles
    return frame


def _hand_energy() -> float:
    a, b = XYZ[0] - XYZ[1], XYZ[2] - XYZ[1]
    theta = math.acos(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    r13 = np.linalg.norm(XYZ[0] - XYZ[2])
    return K * (theta - math.radians(THETA0)) ** 2 + K_UB * (r13 - R_UB) ** 2


def test_an_angle_style_named_charmm_carries_the_four_params() -> None:
    style = _ub_ff().get_style("angle", "charmm")
    assert isinstance(style, mp.ff.forcefield.AngleStyle)
    (t,) = style.get_types()
    assert (t["k"], t["theta0"], t["k_ub"], t["r_ub"]) == (K, THETA0, K_UB, R_UB)


def test_it_compiles_to_the_lammps_energy() -> None:
    frame = _frame()
    compiler = mp.ff.compile.PotentialCompiler(_ub_ff())
    e = compiler.compile(frame).evaluate(frame)[0]
    assert math.isclose(e, _hand_energy(), rel_tol=1e-12)
    assert len(compiler.compile_typed(frame)) == 1


def test_it_round_trips_through_the_section_and_a_store(tmp_path: Path) -> None:
    ff = _ub_ff()
    table = mp.io.mrec.ForceFieldSection.from_forcefield(ff).table("angle", "charmm")
    assert list(table["r_ub"]) == [R_UB]
    path = tmp_path / "ff.mrec"
    mp.io.write_mrec(path, forcefield=ff)
    back = mp.io.mrec.MrecFile(path).forcefield().to_forcefield()
    (t,) = back.get_types("angle")
    assert (t["k"], t["theta0"], t["k_ub"], t["r_ub"]) == (K, THETA0, K_UB, R_UB)
    frame = _frame()
    assert (
        mp.ff.compile.PotentialCompiler(back).compile(frame).evaluate(frame)[0]
        == mp.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)[0]
    )


def test_the_lammps_writer_and_reader_keep_it_as_written(tmp_path: Path) -> None:
    text = _lammps_include(_ub_ff(), _frame())
    assert "angle_style charmm\n" in text
    assert "angle_coeff HA-CT-CT 33.430000 110.100000 22.530000 2.179000\n" in text
    path = tmp_path / "ub.ff"
    path.write_text(text, encoding="utf-8")
    back = mp.io.lammps.LammpsForcefieldReader().read(path)
    (t,) = back.get_types("angle")
    assert (t["k"], t["theta0"], t["k_ub"], t["r_ub"]) == (K, THETA0, K_UB, R_UB)
    data = _lammps_data_coeffs(_ub_ff(), _frame())
    assert "Angle Coeffs # charmm" in data
