import pytest
import molpy as mp

class TestMol2Reader:

    def test_read(self, test_data_path):
        frame = mp.Frame()
        mol2 = mp.io.read_mol2(test_data_path / "data/mol2/ethane.mol2", frame)
        atoms = mol2["atoms"]
        assert atoms.sizes["index"] == 8
        row0 = atoms.isel(index=0)
        assert row0["name"].item() == "C"
        assert tuple(row0["xyz"].values) == pytest.approx((3.1080, 0.6530, -8.5260))
        assert row0["type"].item() == "c3"
        assert int(row0["subst_id"]) == 1
        assert row0["subst_name"].item() == "ETH"
        assert row0["charge"].item() == -0.094100
