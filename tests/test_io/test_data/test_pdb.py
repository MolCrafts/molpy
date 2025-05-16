import molpy as mp

class TestPDBReader:

    def test_read_pdb(self, test_data_dir):
        
        frame = mp.io.read_pdb(test_data_dir )