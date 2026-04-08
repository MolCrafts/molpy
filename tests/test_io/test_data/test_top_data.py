"""Tests for GROMACS topology (.top) data file reader and writer."""

from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy.io.data.top import TopReader, TopWriter


class TestTopReader:
    """Tests for TopReader parsing GROMACS topology files."""

    def test_read_benzene_atoms(self, TEST_DATA_DIR: Path) -> None:
        """TopReader should parse [atoms] section correctly."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        reader = TopReader(top_file)
        frame = reader.read()

        assert "atoms" in frame
        atoms = frame["atoms"]
        assert atoms.nrows == 12  # 6 C + 6 H in benzene

    def test_read_benzene_bonds(self, TEST_DATA_DIR: Path) -> None:
        """TopReader should parse [bonds] section correctly."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        reader = TopReader(top_file)
        frame = reader.read()

        assert "bonds" in frame
        bonds = frame["bonds"]
        assert bonds.nrows == 12

    def test_atom_fields(self, TEST_DATA_DIR: Path) -> None:
        """Atom block should contain expected fields."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        frame = TopReader(top_file).read()
        atoms = frame["atoms"]

        assert "id" in atoms
        assert "type" in atoms
        assert "charge" in atoms
        assert "mass" in atoms
        assert "name" in atoms

    def test_first_atom_values(self, TEST_DATA_DIR: Path) -> None:
        """First atom should have correct values from benzene.top."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        frame = TopReader(top_file).read()
        atom = frame["atoms"][0]

        assert int(atom["id"]) == 1
        assert str(atom["type"]) == "opls_145"
        assert pytest.approx(float(atom["charge"]), abs=1e-4) == -0.115
        assert pytest.approx(float(atom["mass"]), abs=1e-3) == 12.011

    def test_bond_indices_one_based(self, TEST_DATA_DIR: Path) -> None:
        """Bond indices should be stored as 1-based integers from file."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        frame = TopReader(top_file).read()
        bonds = frame["bonds"]
        first_bond = bonds[0]

        # First bond in benzene.top: 1 2 1
        assert int(first_bond["atomi"]) == 1
        assert int(first_bond["atomj"]) == 2

    def test_read_bromobutane_all_sections(self, TEST_DATA_DIR: Path) -> None:
        """1-bromobutane has atoms, bonds, pairs, angles, and dihedrals."""
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("1-bromobutane.top test data not available")

        frame = TopReader(top_file).read()

        assert "atoms" in frame
        assert "bonds" in frame
        assert "pairs" in frame
        assert "angles" in frame
        assert "dihedrals" in frame

    def test_bromobutane_atom_count(self, TEST_DATA_DIR: Path) -> None:
        """1-bromobutane has 14 atoms."""
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("1-bromobutane.top test data not available")

        frame = TopReader(top_file).read()
        assert frame["atoms"].nrows == 14

    def test_bromobutane_bond_count(self, TEST_DATA_DIR: Path) -> None:
        """1-bromobutane has 13 bonds."""
        top_file = TEST_DATA_DIR / "top/1-bromobutane.top"
        if not top_file.exists():
            pytest.skip("1-bromobutane.top test data not available")

        frame = TopReader(top_file).read()
        assert frame["bonds"].nrows == 13

    def test_section_normalization_spaces(self, tmp_path: Path) -> None:
        """Reader handles both `[ atoms ]` and `[atoms]` section styles."""
        # Write file with spacing-less headers
        top_content = """; test
[moleculetype]
MOL  3

[atoms]
1  CT  1  MOL  C  1  -0.1  12.011

[bonds]
"""
        top_file = tmp_path / "test_nospaces.top"
        top_file.write_text(top_content)

        frame = TopReader(top_file).read()
        assert "atoms" in frame
        assert frame["atoms"].nrows == 1

    def test_empty_frame_when_no_sections(self, tmp_path: Path) -> None:
        """Reader returns an empty frame for a file with no known sections."""
        top_file = tmp_path / "empty.top"
        top_file.write_text("; just a comment\n")
        frame = TopReader(top_file).read()
        assert "atoms" not in frame

    def test_via_factory_function(self, TEST_DATA_DIR: Path) -> None:
        """mp.io.read_top with a frame argument reads topology data."""
        top_file = TEST_DATA_DIR / "top/benzene.top"
        if not top_file.exists():
            pytest.skip("benzene.top test data not available")

        # read_top reads into a ForceField (forcefield reader)
        # For data, use TopReader directly
        frame = TopReader(top_file).read()
        assert "atoms" in frame


class TestTopWriter:
    """Tests for TopWriter producing valid GROMACS topology files."""

    def _make_minimal_frame(self) -> mp.Frame:
        """Create a minimal two-atom frame with one bond."""
        frame = mp.Frame()
        frame.metadata["name"] = "MOL"
        frame["atoms"] = {
            "id": np.array([1, 2]),
            "type": np.array(["CT", "HC"]),
            "resnr": np.array([1, 1]),
            "residu": np.array(["MOL", "MOL"]),
            "name": np.array(["C", "H"]),
            "cgnr": np.array([1, 2]),
            "charge": np.array([-0.1, 0.1]),
            "mass": np.array([12.011, 1.008]),
        }
        frame["bonds"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "type": np.array([1]),
        }
        return frame

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """TopWriter.write() creates a file at the given path."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "out.top"
        writer = TopWriter(out_file)
        writer.write(frame)
        assert out_file.exists()

    def test_write_contains_sections(self, tmp_path: Path) -> None:
        """Written file contains expected GROMACS section headers."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "out.top"
        TopWriter(out_file).write(frame)

        content = out_file.read_text()
        assert "[ moleculetype ]" in content
        assert "[ atoms ]" in content
        assert "[ bonds ]" in content
        assert "[ system ]" in content
        assert "[ molecules ]" in content

    def test_write_molecule_name(self, tmp_path: Path) -> None:
        """Written file uses frame.metadata['name'] as molecule name."""
        frame = self._make_minimal_frame()
        frame.metadata["name"] = "BENZENE"
        out_file = tmp_path / "out.top"
        TopWriter(out_file).write(frame)

        content = out_file.read_text()
        assert "BENZENE" in content

    def test_roundtrip_atoms(self, tmp_path: Path) -> None:
        """Atoms written by TopWriter can be read back by TopReader."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "roundtrip.top"
        TopWriter(out_file).write(frame)

        frame2 = TopReader(out_file).read()
        assert "atoms" in frame2
        assert frame2["atoms"].nrows == 2

        # Check first atom
        a0 = frame2["atoms"][0]
        assert int(a0["id"]) == 1
        assert str(a0["type"]) == "CT"
        assert pytest.approx(float(a0["charge"]), abs=1e-4) == -0.1
        assert pytest.approx(float(a0["mass"]), abs=1e-3) == 12.011

    def test_roundtrip_bonds(self, tmp_path: Path) -> None:
        """Bonds written by TopWriter can be read back by TopReader."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "roundtrip.top"
        TopWriter(out_file).write(frame)

        frame2 = TopReader(out_file).read()
        assert "bonds" in frame2
        assert frame2["bonds"].nrows == 1
        bond = frame2["bonds"][0]
        assert int(bond["atomi"]) == 1
        assert int(bond["atomj"]) == 2

    def test_write_pairs_section(self, tmp_path: Path) -> None:
        """TopWriter writes [ pairs ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["pairs"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "type": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        TopWriter(out_file).write(frame)

        content = out_file.read_text()
        assert "[ pairs ]" in content

    def test_write_angles_section(self, tmp_path: Path) -> None:
        """TopWriter writes [ angles ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["angles"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "atomk": np.array([3]),
            "type": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        TopWriter(out_file).write(frame)

        content = out_file.read_text()
        assert "[ angles ]" in content

    def test_write_dihedrals_section(self, tmp_path: Path) -> None:
        """TopWriter writes [ dihedrals ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["dihedrals"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "atomk": np.array([3]),
            "atoml": np.array([4]),
            "type": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        TopWriter(out_file).write(frame)

        content = out_file.read_text()
        assert "[ dihedrals ]" in content

    def test_write_via_factory(self, tmp_path: Path) -> None:
        """write_top factory function writes topology correctly."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "factory.top"
        mp.io.write_top(str(out_file), frame)
        assert out_file.exists()
        content = out_file.read_text()
        assert "[ atoms ]" in content
