import pytest
import molpy as mp

class TestGMXGroReader:

    def test_gro(self, test_data_path):
        fpath = test_data_path / "data/gro/cod_4020641.gro"
        frame = mp.io.read_gro(fpath, frame=mp.Frame())

        assert frame["atoms"].array_length == 81
        atom0 = frame["atoms"][0]
        assert atom0["res_number"] == "1"
        assert atom0["res_name"] == "LIG"
        assert atom0["name"] == "S"
        assert atom0["atomic_number"] == 1
        assert atom0["xyz"] == pytest.approx([0.310,   0.862,   1.316], rel=1e-5)