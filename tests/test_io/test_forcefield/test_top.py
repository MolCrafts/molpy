import pytest
import molpy as mp

class TestGMXTopReader:
    
    def test_read(self, test_data_path):

        ff = mp.io.read_top(
            test_data_path / "forcefield/gromacs/1-bromobutane.top", mp.ForceField())
        
        atomstyle = ff.get_atomstyle("full")
        atomtypes = atomstyle.get_types()
        assert len(atomtypes) == 14
        at135 = atomstyle.get("opls_135")
        assert at135["atom"] == "C"
        assert at135["charge"] == "-0.18"
        assert at135["mass"] == "12.011"

        bondstyle = ff.get_bondstyle("harmonic")
        bondtypes = bondstyle.get_types()
        assert len(bondtypes) == 13

        anglestyle = ff.get_anglestyle("harmonic")
        angletypes = anglestyle.get_types()
        assert len(angletypes) == 24

        diestyle = ff.get_dihedralstyle("harmonic")
        diestypes = diestyle.get_types()
        assert len(diestypes) == 27

        pairstyle = ff.get_pairstyle("lj12-6")
        pairtypes = pairstyle.get_types()
        assert len(pairtypes) == 27