import os
import tempfile
import math

import molpy as mp
from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter


def test_lammps_forcefield_writer_full():
    from molpy import AngleStyle, AtomStyle, BondStyle, DihedralStyle, PairStyle

    ff = mp.ForceField("testff")
    atomstyle = ff.def_style(AtomStyle("full"))
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_H = atomstyle.def_type("H", mass=1.008)

    bondstyle = ff.def_style(BondStyle("harmonic"))
    bondstyle.def_type(atomtype_C, atomtype_H, k=100.0, r0=1.09)

    anglestyle = ff.def_style(AngleStyle("harmonic"))
    anglestyle.def_type(atomtype_C, atomtype_H, atomtype_C, k=50.0, theta0=109.5)

    dihedralstyle = ff.def_style(DihedralStyle("opls"))
    dihedralstyle.def_type(
        atomtype_C, atomtype_H, atomtype_C, atomtype_H, k1=0.5, k2=1.0, k3=0.0, k4=0.0
    )

    pairstyle = ff.def_style(PairStyle("lj/cut"))
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


def test_lammps_forcefield_writer_angle_radians_to_degrees():
    """Test that angle theta0 is correctly converted from radians to degrees.

    Force field stores angles internally in radians, but LAMMPS requires degrees.
    This test verifies the conversion is performed correctly.
    """
    from molpy import AngleStyle, AtomStyle
    from molpy.potential.angle import AngleHarmonicStyle
    import re

    ff = mp.ForceField("test_angle_units")
    atomstyle = ff.def_style(AtomStyle("full"))
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_O = atomstyle.def_type("O", mass=16.00)

    # Create angle style with theta0 in RADIANS (as stored internally)
    theta0_radians = 1.9111355  # ~109.5 degrees in radians
    expected_degrees = math.degrees(theta0_radians)

    anglestyle = ff.def_style(AngleHarmonicStyle())
    anglestyle.def_type(
        atomtype_C, atomtype_O, atomtype_C, k=50.0, theta0=theta0_radians
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "angle_test.ff")
        writer = LAMMPSForceFieldWriter(outpath)
        writer.write(ff)

        with open(outpath) as f:
            content = f.read()

        print(f"Written content:\n{content}")

        # Extract the theta0 value from the angle_coeff line
        match = re.search(r"angle_coeff\s+\S+\s+([\d.]+)\s+([\d.]+)", content)
        assert match, f"Could not find angle_coeff in output: {content}"

        written_k = float(match.group(1))
        written_theta0 = float(match.group(2))

        print(f"Input theta0: {theta0_radians} rad = {expected_degrees}°")
        print(f"Written theta0: {written_theta0}°")

        # Verify the theta0 was converted to degrees
        assert abs(written_theta0 - expected_degrees) < 0.1, (
            f"theta0 not converted correctly: expected ~{expected_degrees:.1f}° "
            f"but got {written_theta0:.1f}°. "
            f"Input was {theta0_radians} rad."
        )

        # Also verify k is preserved
        assert (
            abs(written_k - 50.0) < 0.01
        ), f"k value changed: expected 50.0, got {written_k}"
