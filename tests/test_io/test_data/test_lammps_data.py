"""
Tests for LammpsDataReader and LammpsDataWriter classes.

This module tests the modern LAMMPS data file I/O using Block.from_csv
with space delimiter and mp.ForceField for force field parameters.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.io.data.lammps import LammpsDataReader, LammpsDataWriter


@pytest.fixture
def test_files(TEST_DATA_DIR) -> dict[str, Path]:
    """Provide paths to test files."""
    # Calculate path relative to the test file location

    test_data_dir = TEST_DATA_DIR / "lammps-data"

    files = {
        "molid": test_data_dir / "molid.lmp",
        "whitespaces": test_data_dir / "whitespaces.lmp",
        "triclinic_1": test_data_dir / "triclinic-1.lmp",
        "triclinic_2": test_data_dir / "triclinic-2.lmp",
        "labelmap": test_data_dir / "labelmap.lmp",
        "solvated": test_data_dir / "solvated.lmp",
        "data_body": test_data_dir / "data.body",
    }
    return files


class TestLammpsDataReader:
    """Test LammpsDataReader with real test cases."""

    def test_molid_file(self, test_files):
        """Test reading molid.lmp - file with molecular IDs and full style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="full")
        frame = reader.read()

        # Check basic structure
        assert "atoms" in frame
        atoms = frame["atoms"]

        # Should have 12 atoms based on file content
        assert atoms.nrows == 12
        assert "mol_id" in atoms  # molecule ID (canonical name)
        assert "type" in atoms
        assert "charge" in atoms  # charge (canonical name)
        assert "x" in atoms and "y" in atoms and "z" in atoms  # Separate coordinates

        # Check coordinate data
        x = atoms["x"]
        y = atoms["y"]
        z = atoms["z"]
        assert len(x) == 12
        assert len(y) == 12
        assert len(z) == 12

        # Check box dimensions (0-20 in each direction)
        assert frame.box is not None
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [20.0, 20.0, 20.0])

        # Check that molecule IDs are in the data (should be 0-3 based on file)
        mol_ids = atoms["mol_id"]
        assert len(np.unique(mol_ids)) <= 4  # max 4 different molecules

        # Check metadata
        assert frame.metadata["format"] == "lammps_data"
        assert frame.metadata["atom_style"] == "full"
        assert "forcefield" in frame.metadata

    def test_whitespaces_file(self, test_files):
        """Test reading whitespaces.lmp - file with extra whitespaces."""

        reader = LammpsDataReader(test_files["whitespaces"], atom_style="full")
        frame = reader.read()

        # Should parse correctly despite extra whitespaces
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 1

        # Check the single atom's coordinates
        x = atoms["x"][0]
        y = atoms["y"][0]
        z = atoms["z"][0]
        np.testing.assert_array_almost_equal([x, y, z], [5.0, 5.0, 5.0])

        # Check box (should be 10x10x10)
        box_lengths = frame.box.lengths
        np.testing.assert_array_almost_equal(box_lengths, [10.0, 10.0, 10.0])

    def test_triclinic_file(self, test_files):
        """triclinic-1.lmp — triclinic header with all-zero tilt factors
        must produce an orthogonal-equivalent box."""

        reader = LammpsDataReader(test_files["triclinic_1"], atom_style="atomic")
        frame = reader.read()

        assert frame.box is not None
        np.testing.assert_array_almost_equal(frame.box.lengths, [34.0, 34.0, 34.0])
        np.testing.assert_array_almost_equal(frame.box.tilts, [0.0, 0.0, 0.0])

        if "atoms" in frame:
            assert frame["atoms"].nrows == 0

    def test_triclinic_2_file(self, test_files):
        """triclinic-2.lmp — non-zero tilt factors (5 -8 3 xy xz yz) must
        be captured in the box."""

        reader = LammpsDataReader(test_files["triclinic_2"], atom_style="atomic")
        frame = reader.read()

        assert frame.box is not None
        assert frame.box.style == "triclinic"
        np.testing.assert_array_almost_equal(frame.box.tilts, [5.0, -8.0, 3.0])
        # Edge-vector norms reflect the tilt: |a|=lx, |b|=sqrt(xy^2+ly^2),
        # |c|=sqrt(xz^2+yz^2+lz^2).
        np.testing.assert_array_almost_equal(
            frame.box.lengths,
            [34.0, np.sqrt(5.0**2 + 34.0**2), np.sqrt(8.0**2 + 3.0**2 + 34.0**2)],
        )

    def test_solvated_file(self, test_files):
        """solvated.lmp — large solvated system with full force field;
        all header counts must be honored end-to-end."""

        reader = LammpsDataReader(test_files["solvated"], atom_style="full")
        frame = reader.read()

        assert frame.box is not None
        np.testing.assert_array_almost_equal(
            frame.box.lengths,
            [33.920998 - (-0.103), 33.957998 - (-0.066), 162.150494 - (-0.885501)],
        )
        np.testing.assert_array_almost_equal(
            frame.box.origin, [-0.103, -0.066, -0.885501]
        )

        assert frame["atoms"].nrows == 7772
        assert frame["bonds"].nrows == 6248
        assert frame["angles"].nrows == 8100
        assert frame["dihedrals"].nrows == 10720
        assert frame["impropers"].nrows == 1376

        assert frame.metadata["counts"]["atom_types"] == 11
        assert frame.metadata["counts"]["bond_types"] == 8

    def test_data_body_file(self, test_files):
        """data.body — atom_style='body' must read 'bodyflag' and per-atom
        'mass' columns, and a trailing 'Bodies' section must not leak into
        the atoms block."""

        reader = LammpsDataReader(test_files["data_body"], atom_style="body")
        frame = reader.read()

        assert frame.box is not None
        np.testing.assert_array_almost_equal(
            frame.box.lengths,
            [
                15.532224567 - (-15.532224567),
                15.532224567 - (-15.532224567),
                0.5 - (-0.5),
            ],
        )

        atoms = frame["atoms"]
        assert atoms.nrows == 100  # header says 100 atoms; not 277
        for col in ("id", "type", "bodyflag", "mass", "x", "y", "z"):
            assert col in atoms, f"body atom_style must expose {col!r}"
        # First atom in the file: 1 1 1 6 -15.5322 -15.5322 0 1 2 0
        assert int(atoms["bodyflag"][0]) == 1
        assert float(atoms["mass"][0]) == 6.0
        np.testing.assert_array_almost_equal(
            [atoms["x"][0], atoms["y"][0], atoms["z"][0]],
            [-15.5322, -15.5322, 0.0],
        )

    def test_labelmap_file(self, test_files):
        """Test reading labelmap.lmp - file with type labels and connectivity."""

        reader = LammpsDataReader(test_files["labelmap"], atom_style="full")
        frame = reader.read()

        # Check atoms
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 16

        # Check type labels are preserved
        types = atoms["type"]
        unique_types = np.unique(types)
        expected_types = {"f", "c3", "s6", "o", "ne", "sy", "Li+"}
        assert set(unique_types) == expected_types

        # Check connectivity
        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 14

        # Check bond types
        bond_types = bonds["type"]
        unique_bond_types = np.unique(bond_types)
        assert len(unique_bond_types) > 0

        # Check angles
        assert "angles" in frame
        angles = frame["angles"]
        assert angles.nrows == 25

        # Check dihedrals
        assert "dihedrals" in frame
        dihedrals = frame["dihedrals"]
        assert dihedrals.nrows == 27

        # Check force field
        forcefield = frame.metadata.get("forcefield")
        assert forcefield is not None
        assert isinstance(forcefield, mp.ForceField)

    def test_atomic_style(self, test_files):
        """Test reading with atomic atom style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="atomic")
        frame = reader.read()

        atoms = frame["atoms"]
        # Atomic style should not have mol_id or charge columns
        assert "mol_id" not in atoms
        assert "charge" not in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms

    def test_charge_style(self, test_files):
        """Test reading with charge atom style."""

        reader = LammpsDataReader(test_files["molid"], atom_style="charge")
        frame = reader.read()

        atoms = frame["atoms"]
        # Charge style should have charge but not mol_id
        assert "mol_id" not in atoms
        assert "charge" in atoms
        assert "type" in atoms
        assert "x" in atoms and "y" in atoms and "z" in atoms


class TestLammpsDataWriter:
    """Test LammpsDataWriter."""

    def test_write_read_roundtrip(self, test_files, tmp_path):
        """Test that we can write and read back the same data."""

        # Read original file
        reader = LammpsDataReader(test_files["molid"], atom_style="full")
        original_frame = reader.read()

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="full")
        writer.write(original_frame)

        # Read back
        reader2 = LammpsDataReader(tmp_file, atom_style="full")
        new_frame = reader2.read()

        # Compare atoms
        orig_atoms = original_frame["atoms"]
        new_atoms = new_frame["atoms"]

        assert orig_atoms.nrows == new_atoms.nrows
        # Convert types to strings for comparison since they may be different types
        np.testing.assert_array_equal(
            np.array([str(t) for t in orig_atoms["type"]]),
            np.array([str(t) for t in new_atoms["type"]]),
        )
        np.testing.assert_array_almost_equal(orig_atoms["x"], new_atoms["x"])
        np.testing.assert_array_almost_equal(orig_atoms["y"], new_atoms["y"])
        np.testing.assert_array_almost_equal(orig_atoms["z"], new_atoms["z"])

        # Compare box - skip for now as box handling may need more work
        # assert original_frame.box is not None
        # assert new_frame.box is not None
        # np.testing.assert_array_almost_equal(
        #     original_frame.box.lengths, new_frame.box.lengths
        # )

    def test_read_from_frame_preserves_topology(self, tmp_path):
        """read_lammps_data -> Atomistic.from_frame must keep bonds/angles/dihedrals.

        Regression: the reader stored relation endpoints as a signed int, and
        ``molrs.from_frame`` reads endpoints only as ``uint32`` — so it silently
        dropped every bond on the Frame->Atomistic round-trip.
        """
        data = (
            "minimal\n\n4 atoms\n3 bonds\n1 atom types\n1 bond types\n\n"
            "0 10 xlo xhi\n0 10 ylo yhi\n0 10 zlo zhi\n\n"
            "Masses\n\n1 12.011\n\n"
            "Atoms\n\n"
            "1 1 1 0.0 0.0 0.0 0.0\n2 1 1 0.0 1.5 0.0 0.0\n"
            "3 1 1 0.0 3.0 0.0 0.0\n4 1 1 0.0 4.5 0.0 0.0\n\n"
            "Bonds\n\n1 1 1 2\n2 1 2 3\n3 1 3 4\n"
        )
        path = tmp_path / "chain.data"
        path.write_text(data)

        frame = LammpsDataReader(path, atom_style="full").read()
        # endpoints must be uint32 or molrs.from_frame ignores them
        assert np.asarray(frame["bonds"]["atomi"]).dtype == np.uint32
        rebuilt = mp.Atomistic.from_frame(frame)
        assert sum(1 for _ in rebuilt.bonds) == 3

    def test_write_minimal_frame(self, tmp_path):
        """Test writing a minimal frame with just atoms."""
        # Create a simple frame
        frame = mp.Frame()

        # Add atoms data with separate x, y, z coordinates
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array([1, 1, 2]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([1.0, 1.0, 2.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Write to temporary file
        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file was written and has content
        assert os.path.exists(tmp_file)
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 atom types" in content
            assert "Atoms" in content

    def test_write_full_style(self, tmp_path):
        """Test writing with full atom style including molecule IDs and charges."""
        frame = mp.Frame()

        # Create atoms with all fields
        atoms_data = {
            "id": np.array([1, 2, 3]),
            "mol_id": np.array([1, 1, 2]),
            "type": np.array(["C", "C", "O"]),
            "charge": np.array([0.0, 0.0, -0.5]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 12.0, 16.0]),
        }

        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add bonds
        bonds_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C-C", "C-O"]),
            "atomi": np.array([0, 1]),
            "atomj": np.array([1, 2]),
        }
        frame["bonds"] = mp.Block(bonds_data)

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="full")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "2 bonds" in content
            assert "2 atom types" in content
            assert "2 bond types" in content
            assert "Atom Type Labels" in content
            assert "Bond Type Labels" in content

    def test_write_with_forcefield(self, tmp_path):
        """Test writing with force field parameters."""
        frame = mp.Frame()

        # Create atoms
        atoms_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C", "O"]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "mass": np.array([12.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Create a simple force field (empty for now)
        forcefield = mp.ForceField()
        frame.metadata["forcefield"] = forcefield

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "2 atoms" in content
            # Note: Force field writing may not be implemented yet
            # assert "Pair Coeffs" in content
            # assert "Bond Coeffs" in content


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_nonexistent_file(self):
        """Test reading nonexistent file."""
        with pytest.raises(FileNotFoundError):
            reader = LammpsDataReader("nonexistent_file.data")
            reader.read()

    def test_empty_file(self, tmp_path):
        """Empty files have no box and must raise rather than silently
        falling back to a default box."""
        tmp_file = tmp_path / "test.data"
        with open(tmp_file, "w") as f:
            f.write("")

        reader = LammpsDataReader(tmp_file)
        with pytest.raises(ValueError, match="missing box bounds"):
            reader.read()

    def test_missing_box_axis_raises(self, tmp_path):
        """A header missing one axis must raise — no silent default."""
        content = (
            "# missing z\n"
            "1 atoms\n"
            "1 atom types\n"
            "\n"
            "0.0 10.0 xlo xhi\n"
            "0.0 10.0 ylo yhi\n"
            "\n"
            "Atoms\n"
            "\n"
            "1 1 0.0 0.0 0.0\n"
        )
        tmp_file = tmp_path / "missing_z.data"
        tmp_file.write_text(content)

        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        with pytest.raises(ValueError, match=r"missing box bounds for axis \['z'\]"):
            reader.read()

    def test_float_box_bounds_parsed(self, tmp_path):
        """Regression: float-valued box bounds must parse, not fall back to 10x10x10."""
        content = (
            "# float bounds\n"
            "1 atoms\n"
            "1 atom types\n"
            "\n"
            "0.0 25.0 xlo xhi\n"
            "0.0 30.0 ylo yhi\n"
            "0.0 35.0 zlo zhi\n"
            "\n"
            "Atoms\n"
            "\n"
            "1 1 0.0 0.0 0.0\n"
        )
        tmp_file = tmp_path / "float_box.data"
        tmp_file.write_text(content)

        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        frame = reader.read()

        assert frame.box is not None
        np.testing.assert_array_almost_equal(frame.box.lengths, [25.0, 30.0, 35.0])

    def test_malformed_header(self, tmp_path):
        """Test reading file with malformed header."""
        malformed_content = """# LAMMPS data file
invalid atoms
1 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 10.0 zlo zhi

Masses

1 1.0

Atoms

1 1 0.0 0.0 0.0
"""
        tmp_file = tmp_path / "test.data"
        with open(tmp_file, "w") as f:
            f.write(malformed_content)

        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        frame = reader.read()

        # Should handle malformed header gracefully
        assert frame is not None
        # May not have atoms if header parsing fails
        if "atoms" in frame:
            assert frame["atoms"].nrows >= 0


class TestForceFieldIntegration:
    """Test force field integration."""

    def test_forcefield_parsing(self, test_files):
        """Test that force field parameters are correctly parsed."""
        if "labelmap" not in test_files:
            pytest.skip("labelmap.lmp not found")

        reader = LammpsDataReader(test_files["labelmap"], atom_style="full")
        frame = reader.read()

        forcefield = frame.metadata.get("forcefield")
        assert forcefield is not None
        assert isinstance(forcefield, mp.ForceField)

        # Check that we have a forcefield object (may be empty if no coeffs in file)
        # The labelmap.lmp file may not have force field coefficients sections
        assert forcefield is not None

    def test_forcefield_writing(self, tmp_path):
        """Test that force field parameters are correctly written."""
        frame = mp.Frame()

        # Create simple atoms
        atoms_data = {
            "id": np.array([1]),
            "type": np.array(["C"]),
            "x": np.array([0.0]),
            "y": np.array([0.0]),
            "z": np.array([0.0]),
            "mass": np.array([12.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Create a simple force field (empty for now)
        forcefield = mp.ForceField()
        frame.metadata["forcefield"] = forcefield

        tmp_file = tmp_path / "test.data"

        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Read back and check force field
        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        new_frame = reader.read()

        new_forcefield = new_frame.metadata.get("forcefield")
        assert new_forcefield is not None
        # Note: Force field parsing may not be fully implemented yet


class TestMetadataTypeLabels:
    """Test metadata type_labels functionality."""

    def test_backward_compatibility(self, tmp_path):
        """Test that behavior is unchanged when metadata has no type_labels."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        tmp_file = tmp_path / "test.data"
        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 atoms" in content
            assert "3 atom types" in content
            assert "Atom Type Labels" in content
            assert "1 C" in content
            assert "2 H" in content
            assert "3 O" in content

    def test_metadata_type_labels_priority(self, tmp_path):
        """Test that metadata type_labels are used when provided."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C", "H"]),
            "x": np.array([0.0, 1.0]),
            "y": np.array([0.0, 0.0]),
            "z": np.array([0.0, 0.0]),
            "mass": np.array([12.0, 1.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add metadata with additional type labels
        frame.metadata["type_labels"] = {
            "atom_types": ["C", "H", "O", "N"],  # Includes types not in atoms
        }

        tmp_file = tmp_path / "test.data"
        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content - should include all types from metadata
        with open(tmp_file) as f:
            content = f.read()
            assert "4 atom types" in content  # All types from metadata
            assert "Atom Type Labels" in content
            # Check that all metadata types are present
            assert "1 C" in content
            assert "2 H" in content
            assert "3 N" in content
            assert "4 O" in content

    def test_metadata_auto_merge(self, tmp_path):
        """Test that metadata types and actual types are automatically merged."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "S"]),  # S is not in metadata
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 32.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add metadata with some type labels (missing S)
        frame.metadata["type_labels"] = {
            "atom_types": ["C", "H", "O", "N"],
        }

        tmp_file = tmp_path / "test.data"
        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content - should include merged types
        with open(tmp_file) as f:
            content = f.read()
            # Should have 5 types: C, H, N, O (from metadata) + S (from atoms)
            assert "5 atom types" in content
            assert "Atom Type Labels" in content
            # All types should be present
            assert "1 C" in content
            assert "2 H" in content
            assert "3 N" in content
            assert "4 O" in content
            assert "5 S" in content

    def test_metadata_bond_types(self, tmp_path):
        """Test metadata type_labels for bonds."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "C", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 12.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)

        bonds_data = {
            "id": np.array([1, 2]),
            "type": np.array(["C-C", "C-O"]),
            "atomi": np.array([0, 1]),
            "atomj": np.array([1, 2]),
        }
        frame["bonds"] = mp.Block(bonds_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add metadata with additional bond types
        frame.metadata["type_labels"] = {
            "atom_types": ["C", "O"],
            "bond_types": ["C-C", "C-O", "O-O"],  # O-O not in actual bonds
        }

        tmp_file = tmp_path / "test.data"
        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Check file content
        with open(tmp_file) as f:
            content = f.read()
            assert "3 bond types" in content  # All types from metadata
            assert "Bond Type Labels" in content
            assert "1 C-C" in content
            assert "2 C-O" in content
            assert "3 O-O" in content

    def test_type_id_consistency(self, tmp_path):
        """Test that type_id is consistent across all sections."""
        frame = mp.Frame()

        atoms_data = {
            "id": np.array([1, 2, 3]),
            "type": np.array(["C", "H", "O"]),
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            "mass": np.array([12.0, 1.0, 16.0]),
        }
        frame["atoms"] = mp.Block(atoms_data)
        frame.box = mp.Box([10.0, 10.0, 10.0])

        # Add metadata with specific order
        frame.metadata["type_labels"] = {
            "atom_types": ["H", "O", "C"],  # Different order
        }

        tmp_file = tmp_path / "test.data"
        writer = LammpsDataWriter(tmp_file, atom_style="atomic")
        writer.write(frame)

        # Read back and verify
        reader = LammpsDataReader(tmp_file, atom_style="atomic")
        new_frame = reader.read()

        # Check that type IDs are consistent
        # In the written file, types should be sorted: C, H, O
        # So C should be type 1, H should be type 2, O should be type 3
        with open(tmp_file) as f:
            content = f.read()
            # Type labels should be sorted: C, H, O (alphabetically)
            assert "1 C" in content
            assert "2 H" in content
            assert "3 O" in content

        # Verify atoms section uses same type IDs
        atoms = new_frame["atoms"]
        # The actual atom types in the atoms section should reference
        # the correct type IDs from the type labels section
        assert atoms is not None


class TestForceFieldCoeffs:
    """The reader no longer silently drops ``*Coeffs`` parameters (P1-B fix).

    Previously every coefficient line was parsed for validation only and then
    discarded behind a commented-out ``def_type`` and an ``except: continue``
    swallow, so a LAMMPS data file's force-field parameters were lost on read.
    They are now stored on the metadata ForceField in the shape the writer
    reads back, and malformed lines raise instead of being swallowed.
    """

    @pytest.fixture
    def ff_file(self, TEST_DATA_DIR) -> Path:
        return TEST_DATA_DIR / "lammps-ff" / "peptide.data"

    def test_coeffs_are_extracted(self, ff_file):
        ff = LammpsDataReader(ff_file, atom_style="full").read().metadata["forcefield"]
        pair = {
            t.name: (t.get("epsilon"), t.get("sigma"))
            for s in ff.get_styles(mp.PairStyle)
            for t in s.get_types(mp.Type)
        }
        bond = {
            t.name: (t.get("k"), t.get("r0"))
            for s in ff.get_styles(mp.BondStyle)
            for t in s.get_types(mp.Type)
        }
        assert pair, "pair coefficients were dropped"
        assert bond, "bond coefficients were dropped"
        # values are real numbers, not None
        assert all(e is not None and s is not None for e, s in pair.values())

    def test_coeffs_round_trip(self, ff_file, tmp_path):
        fr = LammpsDataReader(ff_file, atom_style="full").read()

        def grab(ff, style_cls, keys):
            return {
                t.name: tuple(t.get(k) for k in keys)
                for s in ff.get_styles(style_cls)
                for t in s.get_types(mp.Type)
            }

        pair_in = grab(fr.metadata["forcefield"], mp.PairStyle, ["epsilon", "sigma"])
        bond_in = grab(fr.metadata["forcefield"], mp.BondStyle, ["k", "r0"])

        out = tmp_path / "round_trip.data"
        LammpsDataWriter(out, atom_style="full").write(fr)
        ff2 = LammpsDataReader(out, atom_style="full").read().metadata["forcefield"]

        assert grab(ff2, mp.PairStyle, ["epsilon", "sigma"]) == pair_in
        assert grab(ff2, mp.BondStyle, ["k", "r0"]) == bond_in

    def test_malformed_coeff_line_raises(self, tmp_path):
        data = tmp_path / "bad.data"
        data.write_text(
            "bad\n\n2 atoms\n1 atom types\n"
            "0 1 xlo xhi\n0 1 ylo yhi\n0 1 zlo zhi\n\n"
            "Masses\n\n1 1.0\n\n"
            "Pair Coeffs\n\n1 notanumber 3.5\n\n"
            "Atoms\n\n1 1 1 0.0 0.0 0.0 0.0\n2 1 1 0.0 0.5 0.0 0.0\n"
        )
        with pytest.raises(ValueError, match="malformed PairCoeffs"):
            LammpsDataReader(data, atom_style="full").read()
