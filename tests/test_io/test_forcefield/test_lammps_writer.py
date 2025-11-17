import os
import tempfile

import molpy as mp
from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter


def test_lammps_forcefield_writer_full():
    from molpy import AngleStyle, AtomStyle, BondStyle, DihedralStyle, PairStyle

    ff = mp.ForceField("testff")
    atomstyle = ff.def_style(AtomStyle, "full")
    atomtype_C = atomstyle.def_type("C", mass=12.01)
    atomtype_H = atomstyle.def_type("H", mass=1.008)

    bondstyle = ff.def_style(BondStyle, "harmonic")
    bondstyle.def_type(atomtype_C, atomtype_H, k=100.0, r0=1.09)

    anglestyle = ff.def_style(AngleStyle, "harmonic")
    anglestyle.def_type(atomtype_C, atomtype_H, atomtype_C, k=50.0, theta0=109.5)

    dihedralstyle = ff.def_style(DihedralStyle, "opls")
    dihedralstyle.def_type(
        atomtype_C, atomtype_H, atomtype_C, atomtype_H, k1=0.5, k2=1.0, k3=0.0, k4=0.0
    )

    pairstyle = ff.def_style(PairStyle, "lj/cut")
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
