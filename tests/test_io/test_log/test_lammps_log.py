import molpy as mp
import pytest
from molpy.io import load_log

class TestLAMMPSLog:

    @pytest.fixture(scope='class')
    def log(self, test_data_path):
        path = test_data_path / 'log/lammps/'
        log = load_log(path / 'thermo_style_default.log', 'lammps')
        log.read()
        return log
    
    def test_version(self, log):
        print(log)
        assert log['version'] == 'LAMMPS (30 Apr 2019)'

    def test_stage(self, log):

        # assert log['n_stages'] == 1
        print(log['stages'][0]['Step'])
        assert log['stages'][0]['Step'].shape == (11, )
