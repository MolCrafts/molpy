import os
import tempfile
import molpy as mp
from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter

def test_lammps_forcefield_writer_full():
    # 构造力场，包含所有类型
    ff = mp.ForceField("testff")
    atomstyle = ff.def_atomstyle("full")
    atomtype_C = atomstyle.def_type("C", parms=[12.01], kwprms={"mass": 12.01})
    atomtype_H = atomstyle.def_type("H", parms=[1.008], kwprms={"mass": 1.008})

    bondstyle = ff.def_bondstyle("harmonic")
    bondstyle.def_type(atomtype_C, atomtype_H, parms=[100.0, 1.09])

    anglestyle = ff.def_anglestyle("harmonic")
    anglestyle.def_type(atomtype_C, atomtype_H, atomtype_C, parms=[50.0, 109.5])

    dihedralstyle = ff.def_dihedralstyle("opls")
    dihedralstyle.def_type(atomtype_C, atomtype_H, atomtype_C, atomtype_H, parms=[0.5, 1.0, 0.0, 0.0])

    pairstyle = ff.def_pairstyle("lj/cut")
    # 自交类型
    pairstyle.def_type(atomtype_C, atomtype_C, parms=[0.2, 3.4])
    pairstyle.def_type(atomtype_H, atomtype_H, parms=[0.05, 2.5])
    # 交叉类型
    pairstyle.def_type(atomtype_C, atomtype_H, parms=[0.1, 3.0])

    # 写出到临时文件
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "lammps.ff")
        writer = LAMMPSForceFieldWriter(outpath)
        writer.write(ff)
        # 检查文件内容
        with open(outpath) as f:
            content = f.read()
            print(content)
        # 基本 style/type
        assert "bond_style harmonic" in content
        assert "angle_style harmonic" in content
        assert "dihedral_style opls" in content
        assert "pair_style lj/cut" in content
        # coeff
        assert "bond_coeff" in content
        assert "angle_coeff" in content
        assert "dihedral_coeff" in content
        assert "pair_coeff" in content
        # atom type
        assert "C" in content and "H" in content
        # bond参数
        assert "bond_coeff" in content
        # angle参数
        assert "50" in content and "109.5" in content
        # dihedral参数
        assert "0.5" in content and "1" in content
        # pair自交
        assert "pair_coeff C" in content and "3.4" in content
        assert "pair_coeff H" in content and "2.5" in content
        # pair交叉
        assert "pair_coeff C H" in content or "pair_coeff H C" in content
