"""``mp.io.read_top`` / ``mp.io.write_top``: GROMACS topology structure."""

from pathlib import Path

import numpy as np
import pytest

import molpy as mp
from molpy import MetaValue


@pytest.fixture
def top_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "top"


class Testread_top:
    """read_top parses a GROMACS topology into per-section blocks.

    Fixtures: benzene.top (12 atoms, 12 bonds, an #include the reader must
    skip) and chain.top (four atoms with bonds, pairs, angles, dihedrals).
    """

    def test_atoms_section_columns_and_values(self, top_dir: Path) -> None:
        atoms = mp.io.read_top(top_dir / "benzene.top")["atoms"]
        assert atoms.nrows == 12
        for key in ("id", "type", "charge", "mass", "name"):
            assert key in atoms
        assert int(atoms["id"][0]) == 1
        assert str(atoms["type"][0]) == "opls_145"
        assert str(atoms["name"][0]) == "C"
        assert float(atoms["charge"][0]) == pytest.approx(-0.115)
        assert float(atoms["mass"][0]) == pytest.approx(12.011)
        assert list(atoms["type"][6:]) == ["opls_146"] * 6

    def test_bond_indices_stay_one_based(self, top_dir: Path) -> None:
        bonds = mp.io.read_top(top_dir / "benzene.top")["bonds"]
        assert bonds.nrows == 12
        assert int(bonds["atomi"][0]) == 1
        assert int(bonds["atomj"][0]) == 2
        assert bonds["atomi"].min() == 1

    def test_every_bonded_section_is_read(self, top_dir: Path) -> None:
        frame = mp.io.read_top(top_dir / "chain.top")
        assert frame["atoms"].nrows == 4
        assert frame["bonds"].nrows == 3
        assert frame["pairs"].nrows == 1
        assert frame["angles"].nrows == 2
        assert frame["dihedrals"].nrows == 1
        dihedral = frame["dihedrals"]
        assert [int(dihedral[k][0]) for k in ("atomi", "atomj", "atomk", "atoml")] == [
            1,
            2,
            3,
            4,
        ]

    def test_section_headers_without_spaces_are_accepted(self, tmp_path: Path) -> None:
        top_file = tmp_path / "nospaces.top"
        top_file.write_text(
            "[moleculetype]\nMOL  3\n\n[atoms]\n1  CT  1  MOL  C  1  -0.1  12.011\n"
        )
        assert mp.io.read_top(top_file)["atoms"].nrows == 1

    def test_empty_frame_when_no_sections(self, tmp_path: Path) -> None:
        top_file = tmp_path / "empty.top"
        top_file.write_text("; just a comment\n")
        assert "atoms" not in mp.io.read_top(top_file)


class Testwrite_top:
    """Tests for write_top producing valid GROMACS topology files."""

    def _make_minimal_frame(self) -> mp.Frame:
        """Create a minimal two-atom frame with one bond."""
        frame = mp.Frame()
        frame.meta = {"name": MetaValue("string", "MOL")}
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
            "type_id": np.array([1]),
        }
        return frame

    def test_write_creates_file(self, tmp_path: Path) -> None:
        """mp.io.write_top creates a file at the given path."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)
        assert out_file.exists()

    def test_write_contains_sections(self, tmp_path: Path) -> None:
        """Written file contains expected GROMACS section headers."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)

        content = out_file.read_text()
        assert "[ moleculetype ]" in content
        assert "[ atoms ]" in content
        assert "[ bonds ]" in content
        assert "[ system ]" in content
        assert "[ molecules ]" in content

    def test_write_molecule_name(self, tmp_path: Path) -> None:
        """Written file uses frame.meta['name'] as molecule name."""
        frame = self._make_minimal_frame()
        frame.meta = {**frame.meta, "name": MetaValue("string", "BENZENE")}
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)

        content = out_file.read_text()
        assert "BENZENE" in content

    def test_roundtrip_atoms(self, tmp_path: Path) -> None:
        """Atoms written by write_top can be read back by read_top."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "roundtrip.top"
        mp.io.write_top(out_file, frame)

        frame2 = mp.io.read_top(out_file)
        assert "atoms" in frame2
        assert frame2["atoms"].nrows == 2

        # Check first atom
        a0 = frame2["atoms"]
        assert int(a0["id"][0]) == 1
        assert str(a0["type"][0]) == "CT"
        assert pytest.approx(float(a0["charge"][0]), abs=1e-4) == -0.1
        assert pytest.approx(float(a0["mass"][0]), abs=1e-3) == 12.011

    def test_roundtrip_bonds(self, tmp_path: Path) -> None:
        """Bonds written by write_top can be read back by read_top."""
        frame = self._make_minimal_frame()
        out_file = tmp_path / "roundtrip.top"
        mp.io.write_top(out_file, frame)

        frame2 = mp.io.read_top(out_file)
        assert "bonds" in frame2
        assert frame2["bonds"].nrows == 1
        bond = frame2["bonds"]
        assert int(bond["atomi"][0]) == 1
        assert int(bond["atomj"][0]) == 2

    def test_write_pairs_section(self, tmp_path: Path) -> None:
        """write_top writes [ pairs ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["pairs"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "type_id": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)

        content = out_file.read_text()
        assert "[ pairs ]" in content

    def test_write_angles_section(self, tmp_path: Path) -> None:
        """write_top writes [ angles ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["angles"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "atomk": np.array([3]),
            "type_id": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)

        content = out_file.read_text()
        assert "[ angles ]" in content

    def test_write_dihedrals_section(self, tmp_path: Path) -> None:
        """write_top writes [ dihedrals ] section when present in frame."""
        frame = self._make_minimal_frame()
        frame["dihedrals"] = {
            "atomi": np.array([1]),
            "atomj": np.array([2]),
            "atomk": np.array([3]),
            "atoml": np.array([4]),
            "type_id": np.array([1]),
        }
        out_file = tmp_path / "out.top"
        mp.io.write_top(out_file, frame)

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
