import molpy as mp


class TestPDBReader:

    def test_read_pdb(self, test_data_path):

        frame = mp.io.read_pdb(test_data_path / "data/pdb/1avg.pdb", frame=mp.Frame())
        assert frame["atoms"].array_length == 3730
        assert frame["bonds"].array_length == 7