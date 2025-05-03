import pytest
import molpy as mp

class TestMol2Reader:

    def test_read(self, test_data_path):
        frame = mp.Frame()
        mol2 = (
            mp.io.read_mol2(test_data_path / "data/mol2/ethane.mol2", frame)
        )
        atoms = mol2["atoms"]
        assert atoms.array_length == 8
        assert atoms[0]["name"] == "C"
        assert atoms[0]["xyz"] == pytest.approx((3.1080,    0.6530,   -8.5260))
        assert atoms[0]["type"] == "c3"
        assert atoms[0]["subst_id"] == 1
        assert atoms[0]["subst_name"] == "ETH"
        assert atoms[0]["charge"] == -0.094100