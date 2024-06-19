from pathlib import Path
import molpy as mp

class TestAmber:

    def test_tfsi(self, test_data_path):
        work_dir = Path("tests/tmp")
        parameterize = mp.builder.parameterizer.AmberTool(work_dir)
        struct = mp.io.load_struct(test_data_path / "data/pdb/tfsi.pdb")
        path = parameterize.parameterize(struct, {"-c": "bcc", "-n": "-1"}, )
        struct = mp.io.load_struct(path / f"tfsi_AC_converted.lmp", "LAMMPS Data")
        assert struct.n_atoms == 15