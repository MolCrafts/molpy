import pytest
import molpy as mp

class TestGMXTopReader:
    
    def test_read(self, test_data_path):
        top_file = test_data_path / "forcefield/gromacs/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("gromacs test data not available")
        ff = mp.io.read_top(top_file, mp.ForceField())
        
        atomstyle = ff.get_atomstyle("full")
        assert atomstyle is not None
        atomtypes = atomstyle.get_types()
        assert len(atomtypes) == 14
        at135 = atomstyle.get("opls_135")
        assert at135["atom"] == "C"
        assert at135["charge"] == "-0.18"
        assert at135["mass"] == "12.011"

        bondstyle = ff.get_bondstyle("harmonic")
        assert bondstyle is not None
        bondtypes = bondstyle.get_types()
        print(bondtypes)
        assert len(bondtypes) == 13

        anglestyle = ff.get_anglestyle("harmonic")
        assert anglestyle is not None
        angletypes = anglestyle.get_types()
        assert len(angletypes) == 24

        diestyle = ff.get_dihedralstyle("harmonic")
        assert diestyle is not None
        diestypes = diestyle.get_types()
        assert len(diestypes) == 27

        pairstyle = ff.get_pairstyle("lj12-6")
        assert pairstyle is not None
        pairtypes = pairstyle.get_types()
        assert len(pairtypes) == 27