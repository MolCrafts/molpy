import os
import tempfile

import molpy as mp
from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter
from molpy.io.writers import write_lammps_forcefield


def _ff_with_extra_type():
    """A ForceField whose pair style also carries an unused ``oh`` type — the
    kind of cap artifact region parameterisation leaves in a merged force field."""
    from molpy import AtomStyle, PairStyle

    ff = mp.ForceField("capff")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    tc = atomstyle.def_type("c3", mass=12.01)
    th = atomstyle.def_type("h1", mass=1.008)
    to = atomstyle.def_type("oh", mass=16.00)  # not used by any real atom
    pairstyle = ff.def_style(PairStyle(name="lj/cut"))
    pairstyle.def_type(tc, tc, epsilon=0.2, sigma=3.4)
    pairstyle.def_type(th, th, epsilon=0.05, sigma=2.5)
    pairstyle.def_type(to, to, epsilon=0.09, sigma=3.2)
    return ff


def _frame_c3_h1():
    """A 2-atom frame using only ``c3`` and ``h1`` (no ``oh``)."""
    struct = mp.Atomistic()
    a = struct.def_atom({"element": "C", "type": "c3", "x": 0.0, "y": 0.0, "z": 0.0})
    b = struct.def_atom({"element": "H", "type": "h1", "x": 1.0, "y": 0.0, "z": 0.0})
    struct.def_bond(a, b)
    return struct.to_frame()


def test_ff_writer_frame_filters_unused_pair_coeff():
    """With a ``frame``, a coeff for a type absent from the frame's labelmap
    (``oh``) is dropped — else LAMMPS aborts (`type string oh not in labelmap`)."""
    ff = _ff_with_extra_type()
    frame = _frame_c3_h1()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "filtered.ff")
        write_lammps_forcefield(out, ff, frame=frame)
        content = open(out).read()
    assert "pair_coeff c3" in content
    assert "pair_coeff h1" in content
    assert "oh" not in content  # the unused cap type is filtered out


def test_ff_writer_without_frame_keeps_all_types():
    """No ``frame`` -> no whitelist -> every ff type is emitted (unchanged default)."""
    ff = _ff_with_extra_type()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "all.ff")
        write_lammps_forcefield(out, ff)
        content = open(out).read()
    assert "pair_coeff oh oh" in content


def test_lammps_forcefield_writer_full():
    from molpy import AngleStyle, AtomStyle, BondStyle, DihedralStyle, PairStyle

    ff = mp.ForceField("testff")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_H = atomstyle.def_type("H", mass=1.008)

    bondstyle = ff.def_style(BondStyle(name="harmonic"))
    bondstyle.def_type(atomtype_C, atomtype_H, k=100.0, r0=1.09)

    anglestyle = ff.def_style(AngleStyle(name="harmonic"))
    # theta0 follows the molrs convention — stored in degrees, the same unit
    # LAMMPS expects — so the writer emits it unchanged.
    anglestyle.def_type(atomtype_C, atomtype_H, atomtype_C, k=50.0, theta0=109.5)

    dihedralstyle = ff.def_style(DihedralStyle(name="opls"))
    # OPLS dihedral coefficients are stored as c1-c4 (RB/OPLS convention).
    dihedralstyle.def_type(
        atomtype_C, atomtype_H, atomtype_C, atomtype_H, c1=0.5, c2=1.0, c3=0.0, c4=0.0
    )

    pairstyle = ff.def_style(PairStyle(name="lj/cut"))
    pairstyle.def_type(atomtype_C, atomtype_C, epsilon=0.2, sigma=3.4)
    pairstyle.def_type(atomtype_H, atomtype_H, epsilon=0.05, sigma=2.5)
    pairstyle.def_type(atomtype_C, atomtype_H, epsilon=0.1, sigma=3.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "lammps.ff")
        writer = LAMMPSForceFieldWriter(outpath)
        writer.write(ff)
        with open(outpath) as f:
            content = f.read()
            print(content)
        assert "bond_style harmonic" in content
        assert "angle_style harmonic" in content
        assert "dihedral_style opls" in content
        assert "pair_style lj/cut" in content
        assert "bond_coeff" in content
        assert "angle_coeff" in content
        assert "dihedral_coeff" in content
        assert "pair_coeff" in content
        assert "C" in content and "H" in content
        assert "bond_coeff" in content
        assert "50" in content and "109.5" in content
        assert "0.5" in content and "1" in content
        assert "pair_coeff C" in content and "3.4" in content
        assert "pair_coeff H" in content and "2.5" in content
        assert "pair_coeff C H" in content or "pair_coeff H C" in content


def test_lammps_forcefield_writer_angle_degrees_passthrough():
    """Angle theta0 (molrs convention: degrees) is written through unchanged.

    molrs stores angle equilibria in degrees — the same unit LAMMPS
    ``angle_coeff harmonic`` expects — so the writer must NOT apply a radian
    conversion. Regression guard against the old rad->deg double-conversion
    (e.g. 104.52 deg wrongly emitted as 5988.55).
    """
    import re

    from molpy import AtomStyle
    from molpy.core.forcefield import AngleHarmonicStyle

    ff = mp.ForceField("test_angle_units")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_O = atomstyle.def_type("O", mass=16.00)

    theta0_degrees = 104.52  # e.g. TIP3P H-O-H, in degrees as molrs stores it
    anglestyle = ff.def_style(AngleHarmonicStyle())
    anglestyle.def_type(
        atomtype_C, atomtype_O, atomtype_C, k=50.0, theta0=theta0_degrees
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "angle_test.ff")
        LAMMPSForceFieldWriter(outpath).write(ff)
        with open(outpath) as f:
            content = f.read()

        match = re.search(r"angle_coeff\s+\S+\s+([\d.]+)\s+([\d.]+)", content)
        assert match, f"Could not find angle_coeff in output: {content}"
        written_k = float(match.group(1))
        written_theta0 = float(match.group(2))

        assert abs(written_theta0 - theta0_degrees) < 0.1, (
            f"theta0 should pass through in degrees: expected {theta0_degrees}, "
            f"got {written_theta0}."
        )
        # k is preserved unchanged
        assert abs(written_k - 50.0) < 0.01, f"k value changed: got {written_k}"
