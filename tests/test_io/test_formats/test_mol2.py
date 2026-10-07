"""Tripos MOL2 reader and writer.

The reader maps the ATOM and BOND sections onto canonical atom / bond blocks
(0-based bond indices, ``res_id`` / ``res_name`` from the substructure
columns) and reports an unreadable file as ``OSError``. Fixtures live in ``tests-data/mol2``: ethane (charges,
seven single bonds), li (no charges, empty BOND section), naphthalene (two
fused aromatic rings) and bond_orders (a double, an amide and a single bond).
"""

from pathlib import Path

import numpy as np
import pytest

import molpy as mp


@pytest.fixture
def mol2_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "mol2"


def _read(path: Path) -> mp.Frame:
    return mp.io.read_mol2(path)


class TestReadMol2:
    def test_atom_section_maps_to_canonical_columns(self, mol2_dir):
        atoms = _read(mol2_dir / "ethane.mol2")["atoms"]
        assert atoms.n_rows == 8
        assert list(atoms["name"][:3]) == ["C", "C", "H"]
        assert list(atoms["type"][:3]) == ["c3", "c3", "hc"]
        assert atoms["res_id"].tolist() == [1] * 8
        assert list(atoms["res_name"]) == ["ETH"] * 8
        assert atoms["x", "y", "z"].dtype == np.float64
        np.testing.assert_allclose(atoms["x", "y", "z"][0], [3.108, 0.653, -8.526])
        np.testing.assert_allclose(atoms["charge"][:3], [-0.0941, -0.0941, 0.0317])

    def test_bond_indices_are_zero_based(self, mol2_dir):
        bonds = _read(mol2_dir / "ethane.mol2")["bonds"]
        assert bonds.n_rows == 7
        assert bonds["atomi"].tolist() == [0, 0, 0, 0, 1, 1, 1]
        assert bonds["atomj"].tolist() == [1, 2, 3, 4, 5, 6, 7]

    def test_bond_order_tokens_are_kept_and_decoded(self, mol2_dir):
        bonds = _read(mol2_dir / "bond_orders.mol2")["bonds"]
        assert list(bonds["type"]) == ["2", "am", "1"]
        assert bonds["bond_number"].tolist() == [2, 1, 1]

    def test_ring_closures_reference_earlier_atoms(self, mol2_dir):
        bonds = _read(mol2_dir / "naphthalene.mol2")["bonds"]
        assert bonds.n_rows == 11
        pairs = set(zip(bonds["atomi"].tolist(), bonds["atomj"].tolist()))
        assert {(5, 0), (9, 2), (1, 6)} <= pairs
        assert set(bonds["type"]) == {"ar"}
        assert bonds["bond_number"].tolist() == [0] * 11

    def test_no_charges_and_empty_bond_section(self, mol2_dir):
        frame = _read(mol2_dir / "li.mol2")
        atoms = frame["atoms"]
        assert atoms.n_rows == 1
        assert list(atoms["type"]) == ["opls_404"]
        assert "charge" not in atoms
        assert "bonds" not in frame
        assert frame.meta["title"] == "RES"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(OSError):
            _read(tmp_path / "missing.mol2")

    def test_malformed_coordinate_raises(self, mol2_dir, tmp_path):
        bad = tmp_path / "bad.mol2"
        bad.write_text(
            (mol2_dir / "ethane.mol2")
            .read_text(encoding="utf-8")
            .replace("3.1080", "three"),
            encoding="utf-8",
        )
        with pytest.raises(OSError):
            _read(bad)


class TestWriteMol2:
    def test_round_trip_preserves_atoms_bonds_and_charges(self, mol2_dir, tmp_path):
        frame = _read(mol2_dir / "ethane.mol2")
        out = tmp_path / "out.mol2"
        mp.io.write_mol2(out, frame)
        back = _read(out)
        assert back["atoms"].n_rows == 8
        assert list(back["atoms"]["type"]) == list(frame["atoms"]["type"])
        np.testing.assert_allclose(back["atoms"]["charge"], frame["atoms"]["charge"])
        assert back["bonds"]["atomi"].tolist() == frame["bonds"]["atomi"].tolist()
        assert back["bonds"]["atomj"].tolist() == frame["bonds"]["atomj"].tolist()
