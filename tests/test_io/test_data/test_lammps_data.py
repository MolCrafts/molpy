import pytest
import molpy as mp


class TestReadLammpsData:

    @pytest.fixture()
    def lammps_data(self, test_data_path):
        return test_data_path / "data/lammps-data"

    def test_molid(self, lammps_data):
        
        reader = mp.io.data.LammpsDataReader(lammps_data / "molid.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert frame["atoms"].shape[0] == 12

    def test_labelmap(self, lammps_data):
        
        reader = mp.io.data.LammpsDataReader(lammps_data / "labelmap.lmp")
        frame = mp.Frame()
        frame = reader.read(frame)
        assert len(frame["atoms"]) == 16
        assert "type" in frame["atoms"]
        print(frame["atoms", "type"])
        assert (frame["atoms"]["type"] == ['f', 'c3', 'f', 'f', 's6', 'o', 'o', 'ne', 'sy', 'o', 'o', 'c3', 'f', 'f', 'f', 'Li+']).all()