import pytest
import numpy as np
import molpy as mp
from molpy.io.data.pdb import PDBReader, PDBWriter


@pytest.fixture(scope="module")
def pdb_test_files(TEST_DATA_DIR):
    """Provide paths to PDB test files in chemfile-testcases/data/pdb."""
    base = TEST_DATA_DIR / "data/pdb"
    files = list(base.glob("*.pdb"))
    if not files:
        pytest.skip("No PDB test files found in chemfile-testcases/data/pdb")
    return {f.name: f for f in files}


class TestPDBIO:
    def test_read_1avg(self, pdb_test_files):
        if "1avg.pdb" not in pdb_test_files:
            pytest.skip("1avg.pdb not found")
        reader = PDBReader(str(pdb_test_files["1avg.pdb"]))
        frame = reader.read()
        atoms = frame["atoms"]
        assert atoms["name"].shape[0] == 3730
        assert frame.box is not None
        # Bonds block exists and has correct shape
        bonds = frame["bonds"]
        assert bonds["i"].shape[0] == 7
        assert bonds["j"].shape[0] == 7

    def test_read_water(self, pdb_test_files):
        if "water.pdb" not in pdb_test_files:
            pytest.skip("water.pdb not found")
        reader = PDBReader(str(pdb_test_files["water.pdb"]))
        frame = reader.read()
        atoms = frame["atoms"]
        n_atoms = atoms["name"].shape[0]
        assert n_atoms > 1000
        assert frame.box is not None
        assert atoms["xyz"].shape == (n_atoms, 3)

    def test_write_and_read_roundtrip(self, tmp_path, pdb_test_files):
        if "1avg.pdb" not in pdb_test_files:
            pytest.skip("1avg.pdb not found")
        reader = PDBReader(str(pdb_test_files["1avg.pdb"]))
        frame = reader.read()
        out_path = tmp_path / "roundtrip.pdb"
        writer = PDBWriter(str(out_path))
        writer.write(frame)
        frame2 = PDBReader(str(out_path)).read()
        atoms1 = frame["atoms"]
        atoms2 = frame2["atoms"]
        assert atoms1["name"].shape == atoms2["name"].shape
        assert np.allclose(atoms1["xyz"], atoms2["xyz"])  # allow float tolerance
        assert frame2.box is not None

    def test_missing_fields(self, tmp_path):
        pdb_content = "ATOM      1  X   MOL A   1       0.000   0.000   0.000           X  \nEND\n"
        pdb_path = tmp_path / "min.pdb"
        pdb_path.write_text(pdb_content)
        reader = PDBReader(str(pdb_path))
        frame = reader.read()
        atoms = frame["atoms"]
        assert atoms["name"][0] == "X"
        assert np.allclose(atoms["xyz"][0], [0.0, 0.0, 0.0])
        # box may be None or default
        assert frame.box is None or hasattr(frame.box, "matrix")
