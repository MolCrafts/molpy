import importlib

import numpy as np
import pytest


@pytest.fixture(
    scope="module",
    params=["molpy.io.data.pdb"],
    ids=["molpy"],
)
def pdb_backend(request):
    return importlib.import_module(request.param)


@pytest.fixture(scope="module")
def pdb_test_files(TEST_DATA_DIR):
    """Provide paths to PDB test files in chemfile-testcases/pdb."""
    base = TEST_DATA_DIR / "pdb"
    files = list(base.glob("*.pdb"))
    if not files:
        pytest.skip("No PDB test files found in chemfile-testcases/pdb")
    return {f.name: f for f in files}


class TestPDBIO:
    def test_read_1avg(self, pdb_test_files, pdb_backend):
        if "1avg.pdb" not in pdb_test_files:
            pytest.skip("1avg.pdb not found")
        reader = pdb_backend.PDBReader(pdb_test_files["1avg.pdb"])
        frame = reader.read()
        atoms = frame["atoms"]
        assert atoms["name"].shape[0] == 3730
        assert frame.box is not None
        # Bonds block exists and has correct shape
        bonds = frame["bonds"]
        assert bonds["atomi"].shape[0] == 7
        assert bonds["atomj"].shape[0] == 7

    def test_read_water(self, pdb_test_files, pdb_backend):
        if "water.pdb" not in pdb_test_files:
            pytest.skip("water.pdb not found")
        # Single-frame read returns the first model of a multi-frame PDB.
        reader = pdb_backend.PDBReader(pdb_test_files["water.pdb"])
        frame = reader.read()
        atoms = frame["atoms"]
        n_atoms = atoms["name"].shape[0]
        assert n_atoms > 0
        # Separate x, y, z fields
        assert "x" in atoms and "y" in atoms and "z" in atoms
        assert atoms["x"].shape[0] == n_atoms
        assert atoms["y"].shape[0] == n_atoms
        assert atoms["z"].shape[0] == n_atoms

        # All models are accessible as a trajectory (one frame per model).
        import molpy as mp

        frames = mp.io.read_pdb_trajectory(pdb_test_files["water.pdb"])
        assert len(frames) > 1
        assert all(f["atoms"]["name"].shape[0] == n_atoms for f in frames)

    def test_write_and_read_roundtrip(self, tmp_path, pdb_test_files, pdb_backend):
        if "1avg.pdb" not in pdb_test_files:
            pytest.skip("1avg.pdb not found")
        reader = pdb_backend.PDBReader(pdb_test_files["1avg.pdb"])
        frame = reader.read()
        out_path = tmp_path / "roundtrip.pdb"
        writer = pdb_backend.PDBWriter(out_path)
        writer.write(frame)
        frame2 = pdb_backend.PDBReader(out_path).read()
        atoms1 = frame["atoms"]
        atoms2 = frame2["atoms"]
        assert atoms1["name"].shape == atoms2["name"].shape
        # Check separate x, y, z fields
        assert np.allclose(atoms1["x"], atoms2["x"])
        assert np.allclose(atoms1["y"], atoms2["y"])
        assert np.allclose(atoms1["z"], atoms2["z"])
        assert frame2.box is not None
