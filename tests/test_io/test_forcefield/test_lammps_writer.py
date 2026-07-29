import math
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

    # Force-field params are stored in the molrs internal convention: harmonic
    # stiffness in the ½k(x−x₀)² form (so LAMMPS K = k/2) and angles in radians.
    # The writer inverts both, so K = 100/2 = 50 and theta0 = 109.5° here.
    bondstyle = ff.def_style(BondStyle(name="harmonic"))
    bondstyle.def_type(atomtype_C, atomtype_H, k=100.0, r0=1.09)

    anglestyle = ff.def_style(AngleStyle(name="harmonic"))
    anglestyle.def_type(
        atomtype_C, atomtype_H, atomtype_C, k=100.0, theta0=math.radians(109.5)
    )

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
        assert "50" in content and "109.5" in content  # K = k/2, theta0 in deg
        assert "0.5" in content and "1" in content
        assert "pair_coeff C" in content and "3.4" in content
        assert "pair_coeff H" in content and "2.5" in content
        assert "pair_coeff C H" in content or "pair_coeff H C" in content


def test_lammps_forcefield_writer_angle_units_converted():
    """Angle params convert from molrs internal units to LAMMPS on write.

    molrs stores harmonic stiffness in the ``½k(θ−θ₀)²`` form and ``theta0`` in
    radians; LAMMPS ``angle_coeff harmonic`` wants ``K = k/2`` and degrees. The
    writer must apply both conversions -- guarding both the old missing rad->deg
    conversion and a doubled force constant.
    """
    import re

    from molpy import AtomStyle
    from molpy.core.forcefield import AngleHarmonicStyle

    ff = mp.ForceField("test_angle_units")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_O = atomstyle.def_type("O", mass=16.00)

    theta0_degrees = 104.52  # e.g. TIP3P H-O-H
    anglestyle = ff.def_style(AngleHarmonicStyle())
    anglestyle.def_type(
        atomtype_C, atomtype_O, atomtype_C,
        k=100.0,  # ½k form -> LAMMPS K = 50
        theta0=math.radians(theta0_degrees),  # stored in radians
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
            f"theta0 should be written in degrees: expected {theta0_degrees}, "
            f"got {written_theta0}."
        )
        assert abs(written_k - 50.0) < 0.01, (
            f"K should be k/2 = 50.0, got {written_k}"
        )


def test_lj_plus_coulomb_written_as_combined_style():
    """A split ``lj/cut`` + ``coul/cut`` force field -- the shape molrs returns
    after reading ``lj/cut/coul/long`` -- is written back as the combined
    ``lj/cut/coul/cut`` kernel, not a hybrid.

    Writing it as ``hybrid`` would need a ``pair_coeff * * coul/cut`` wildcard,
    which marks every cross pair as explicitly set and defeats LAMMPS's LJ
    mixing, so off-diagonal pairs lose all repulsion. The combined kernel mixes
    correctly and needs no wildcard.
    """
    from molpy import AtomStyle, PairStyle

    ff = mp.ForceField("split")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    tc = atomstyle.def_type("c3", mass=12.01)
    th = atomstyle.def_type("h1", mass=1.008)
    lj = ff.def_style(PairStyle(name="lj/cut"))
    lj.def_type(tc, tc, epsilon=0.2, sigma=3.4)
    lj.def_type(th, th, epsilon=0.05, sigma=2.5)
    ff.def_style(PairStyle(name="coul/cut"))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "combined.ff")
        LAMMPSForceFieldWriter(out).write(ff)
        content = open(out).read()

    assert "pair_style lj/cut/coul/cut 10.000000 10.000000" in content
    assert "hybrid" not in content
    assert "* *" not in content  # no wildcard that would defeat LJ mixing
    assert "pair_coeff c3 c3 0.200000 3.400000" in content


def test_bonded_coeffs_written_in_lammps_units(tmp_path):
    """Bond, angle and dihedral coefficients are emitted in LAMMPS units.

    Ground-truth check on every bonded conversion. Params are built in the molrs
    internal convention -- harmonic ``k`` in the ``½k`` form (LAMMPS ``K = k/2``),
    angles in radians, and a fourier dihedral under the ``k1``/``n1``/``d1``
    keys the reader uses -- and the writer must invert each: halve the force
    constant, convert angles to degrees, and keep the dihedral periodicity.
    """
    from molpy import AngleStyle, AtomStyle, BondStyle, DihedralStyle

    ff = mp.ForceField("bonded")
    atomstyle = ff.def_style(AtomStyle(name="full"))
    c3 = atomstyle.def_type("c3", mass=12.01)
    oh = atomstyle.def_type("oh", mass=16.00)
    ho = atomstyle.def_type("ho", mass=1.008)

    ff.def_style(BondStyle(name="harmonic")).def_type(c3, c3, k=457.78, r0=1.5354)
    ff.def_style(AngleStyle(name="harmonic")).def_type(
        c3, c3, oh, k=153.58, theta0=math.radians(109.66)
    )
    ff.def_style(DihedralStyle(name="fourier")).def_type(
        c3, c3, oh, ho, k1=0.06, n1=3.0, d1=0.0
    )

    out = tmp_path / "bonded.ff"
    LAMMPSForceFieldWriter(str(out)).write(ff)
    written = {
        line.split()[0]: tuple(round(float(x), 4) for x in line.split()[2:])
        for line in out.read_text().splitlines()
        if line.split()[:1]
        and line.split()[0] in ("bond_coeff", "angle_coeff", "dihedral_coeff")
    }
    assert written["bond_coeff"] == (228.89, 1.5354)  # K = k/2, not 2K
    assert written["angle_coeff"] == (76.79, 109.66)  # K = k/2, theta0 in degrees
    assert written["dihedral_coeff"] == (1.0, 0.06, 3.0, 0.0)  # n = 3 preserved
