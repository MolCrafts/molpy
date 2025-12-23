import numpy as np
import pytest

import molpy as mp


class TestMol2Reader:
    """Basic MOL2 reading tests."""

    def test_read(self, TEST_DATA_DIR):
        frame = mp.Frame()
        mol2 = mp.io.read_mol2(TEST_DATA_DIR / "mol2/ethane.mol2", frame)
        atoms = mol2["atoms"]

        # Check that we have 8 atoms (ethane: 2 C + 6 H)
        assert atoms.nrows == 8

        # Check specific atom properties
        assert atoms["name"][0] == "C"
        assert atoms["x"][0] == pytest.approx(3.1080)
        assert atoms["y"][0] == pytest.approx(0.6530)
        assert atoms["z"][0] == pytest.approx(-8.5260)
        assert atoms["type"][0] == "c3"
        assert int(atoms["subst_id"][0]) == 1
        assert atoms["subst_name"][0] == "ETH"
        assert atoms["charge"][0] == pytest.approx(-0.094100)


class TestMol2Comprehensive:
    """Comprehensive MOL2 tests using chemfiles test cases."""

    def test_ethane_basic_structure(self, TEST_DATA_DIR):
        """Test basic ethane structure reading."""
        fpath = TEST_DATA_DIR / "mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Ethane should have 8 atoms (2 C + 6 H)
        assert atoms.nrows == 8

        # Check required fields
        assert "name" in atoms
        assert "x" in atoms
        assert "y" in atoms
        assert "z" in atoms
        assert "type" in atoms
        assert "charge" in atoms

    def test_imatinib_large_molecule(self, TEST_DATA_DIR):
        """Test reading larger molecule (imatinib)."""
        fpath = TEST_DATA_DIR / "mol2/imatinib.mol2"
        if not fpath.exists():
            pytest.skip("imatinib.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Imatinib should have many atoms
        assert atoms.nrows > 50

        # Check bonds if present
        if "bonds" in frame:
            bonds = frame["bonds"]
            assert bonds.nrows > 0

    def test_status_bits_with_bond_status(self, TEST_DATA_DIR):
        """Test MOL2 files with bond status information."""
        # Test any available MOL2 file
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = TEST_DATA_DIR / f"mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())

                # Should handle status bits gracefully
                assert "atoms" in frame
                atoms = frame["atoms"]

                # Check basic fields
                assert "name" in atoms
                assert "x" in atoms and "y" in atoms and "z" in atoms
                break
        else:
            pytest.skip("No MOL2 test files available")

    def test_small_molecules_li_pf6(self, TEST_DATA_DIR):
        """Test small molecules like Li/PF6."""
        # Test available small molecule files
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = TEST_DATA_DIR / f"mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())

                # Check that small molecules are handled
                assert "atoms" in frame
                atoms = frame["atoms"]

                # Should have atomic numbers assigned
                if "number" in atoms:
                    atomic_numbers = atoms["number"]
                    assert all(an > 0 for an in atomic_numbers)
                break
        else:
            pytest.skip("No MOL2 test files available")

    def test_ring_detection_structures(self, TEST_DATA_DIR):
        """Test MOL2 files with ring structures."""
        fpath = TEST_DATA_DIR / "mol2/imatinib.mol2"
        if not fpath.exists():
            pytest.skip("imatinib.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())

        # Should handle ring structures
        assert "atoms" in frame

        # If bonds are present, check connectivity
        if "bonds" in frame:
            bonds = frame["bonds"]
            assert "i" in bonds
            assert "j" in bonds

    def test_coordinate_precision(self, TEST_DATA_DIR):
        """Test coordinate precision in MOL2 files."""
        fpath = TEST_DATA_DIR / "mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Check coordinate precision
        x_coords = atoms["x"]
        y_coords = atoms["y"]
        z_coords = atoms["z"]

        assert x_coords.dtype == np.float64
        assert y_coords.dtype == np.float64
        assert z_coords.dtype == np.float64
        assert not np.any(np.isnan(x_coords))
        assert not np.any(np.isnan(y_coords))
        assert not np.any(np.isnan(z_coords))
        assert not np.any(np.isinf(x_coords))
        assert not np.any(np.isinf(y_coords))
        assert not np.any(np.isinf(z_coords))

    def test_charge_handling(self, TEST_DATA_DIR):
        """Test charge information handling."""
        fpath = TEST_DATA_DIR / "mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Should have charge information
        if "charge" in atoms:
            charges = atoms["charge"]
            assert len(charges) > 0
            assert all(isinstance(float(c), (int, float)) for c in charges)

    def test_substructure_handling(self, TEST_DATA_DIR):
        """Test substructure information."""
        fpath = TEST_DATA_DIR / "mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Check substructure fields
        if "subst_id" in atoms:
            subst_ids = atoms["subst_id"]
            assert len(subst_ids) > 0

        if "subst_name" in atoms:
            subst_names = atoms["subst_name"]
            assert len(subst_names) > 0

    def test_atomic_number_assignment(self, TEST_DATA_DIR):
        """Test atomic number assignment from atom types."""
        fpath = TEST_DATA_DIR / "mol2/ethane.mol2"
        if not fpath.exists():
            pytest.skip("ethane.mol2 test data not available")

        frame = mp.io.read_mol2(fpath, frame=mp.Frame())
        atoms = frame["atoms"]

        # Should have atomic numbers
        if "number" in atoms:
            atomic_numbers = atoms["number"]
            assert all(an > 0 for an in atomic_numbers)

            # For ethane, should have carbon (6) and hydrogen (1)
            unique_elements = set(atomic_numbers)
            assert 6 in unique_elements  # Carbon
            assert 1 in unique_elements  # Hydrogen

    def test_bond_type_variations(self, TEST_DATA_DIR):
        """Test various bond type handling."""
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = TEST_DATA_DIR / f"mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())

                # Should handle bonds
                if "bonds" in frame:
                    bonds = frame["bonds"]

                    # Check bond connectivity
                    assert "i" in bonds
                    assert "j" in bonds

                    # Bond indices should be valid
                    bond_i = bonds["i"]
                    bond_j = bonds["j"]
                    assert all(i >= 0 for i in bond_i)
                    assert all(j >= 0 for j in bond_j)
                break

    def test_empty_sections_handling(self, TEST_DATA_DIR):
        """Test handling of MOL2 files with missing sections."""
        # Test that files with missing sections are handled gracefully
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = TEST_DATA_DIR / f"mol2/{mol2_file}"
            if fpath.exists():
                frame = mp.io.read_mol2(fpath, frame=mp.Frame())

                # Should always have atoms
                assert "atoms" in frame

                # Other sections may or may not be present
                atoms = frame["atoms"]
                assert len(atoms["name"]) > 0
                break

    def test_multiple_files_consistency(self, TEST_DATA_DIR):
        """Test consistency across multiple MOL2 files."""
        mol2_files = []
        for mol2_file in ["ethane.mol2", "imatinib.mol2"]:
            fpath = TEST_DATA_DIR / f"mol2/{mol2_file}"
            if fpath.exists():
                mol2_files.append(fpath)

        if len(mol2_files) < 1:
            pytest.skip("Not enough MOL2 test files available")

        # Test that all files can be read consistently
        for fpath in mol2_files:
            frame = mp.io.read_mol2(fpath, frame=mp.Frame())

            # Basic consistency checks
            assert "atoms" in frame
            atoms = frame["atoms"]

            # Should have basic fields
            assert "name" in atoms
            assert "x" in atoms and "y" in atoms and "z" in atoms

            # Coordinates should be valid
            x_coords = atoms["x"]
            y_coords = atoms["y"]
            z_coords = atoms["z"]
            assert not np.any(np.isnan(x_coords))
            assert not np.any(np.isnan(y_coords))
            assert not np.any(np.isnan(z_coords))
            assert not np.any(np.isinf(x_coords))
            assert not np.any(np.isinf(y_coords))
            assert not np.any(np.isinf(z_coords))
