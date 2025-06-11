import molpy as mp
import pytest

pytest.skip("Typifier tests require builder features not implemented", allow_module_level=True)

class TestTypifier:

    @pytest.fixture(scope='class', name="ethane")
    def ethane(self):
        class Ethane(mp.Struct):
            def __init__(self):
                super().__init__()
                ch3_1 = mp.builder.CH3()
                ch3_1.translate(-ch3_1["atoms"][0].xyz)
                ch3_2 = mp.builder.CH3()
                ch3_2.translate(-ch3_2["atoms"][0].xyz)
                ch3_2.rotate(180, axis=(0, 1, 0))
                self.add_struct(ch3_1)
                self.add_struct(ch3_2)
                self.add_bond(ch3_1["atoms"][0], ch3_2["atoms"][0])

        return Ethane()
    
    @pytest.fixture(scope='class', name="ff_ethane")
    def ff_ethane(self):
        ff = mp.ForceField("ethane")
        atomstyle = ff.def_atomstyle("full")
        C_type = atomstyle.def_type("C", charge=-0.1)
        H_type = atomstyle.def_type("H", charge=0.1)
        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_type(C_type, C_type, p1=1.0)
        bondstyle.def_type(C_type, H_type, p1=2.0)
        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type(C_type, C_type, H_type, p1=109.5)
        anglestyle.def_type(H_type, C_type, H_type, p1=109.5)
        dihedralstyle = ff.def_dihedralstyle("harmonic")
        dihedralstyle.def_type(H_type, C_type, C_type, H_type, p1=180.0)
        return ff

    def test_bond(self, ff_ethane, ethane):

        typifier = mp.typifier.Typifier(ff_ethane)
        typifier.typify_bonds(ethane)
        
        bonds = ethane.bonds
        # assert bonds[0]["style"] == "harmonic"
        assert bonds[0]["type"] == f"C-H"
        assert bonds[1]["type"] == f"C-H"
        assert bonds[2]["type"] == f"C-H"