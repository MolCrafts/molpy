import pytest
import molpy as mp


class TestReadLammpsData:

    @pytest.fixture()
    def lammps_data(self, test_data_path):
        return test_data_path / "data/lammps-data"

    def test_molid(self, lammps_data):
        
        reader = mp.io.data.LammpsDataReader(lammps_data / "molid.lmp")
        system = reader.read(mp.System())
        frame = system.frame
        assert frame["atoms"].shape[0] == 12

    def test_labelmap(self, lammps_data):
        
        reader = mp.io.data.LammpsDataReader(lammps_data / "labelmap.lmp")
        system = reader.read(mp.System())
        frame = system.frame
        assert len(frame["atoms"]) == 16
        assert all(frame["atoms", "type_label"] == ['f', 'c3', 'f', 'f', 's6', 'o', 'o', 'ne', 'sy', 'o', 'o', 'c3', 'f', 'f', 'f', 'Li+'])

        ff = system.forcefield
        assert set([t for t in ff.atomstyles[0].types]) == set(frame["atoms", "type_label"])