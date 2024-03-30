import molpy as mp
import pytest

class TestLAMMPSLog:

    @pytest.fixture(scope='class')
    def log(self, test_data_path):
        path = test_data_path / 'log/lammps/'
        return mp.LAMMPSLog.from_file(path / 'log.lammps')
    
    def test_version(self, log):
        
        assert log.version == 'LAMMPS (29 Oct 2020)'
