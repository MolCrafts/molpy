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

    def test_bond_order_tokens_decode_and_only_the_extra_is_type(self, mol2_dir):
        bonds = _read(mol2_dir / "bond_orders.mol2")["bonds"]
        assert bonds["bond_type"].tolist() == [2, 1, 1]
        assert bonds["bond_number"].tolist() == [2, 1, 1]
        assert list(bonds.validity("type")) == [False, True, False]
        assert bonds["type"][1] == "am"

    def test_ring_closures_reference_earlier_atoms(self, mol2_dir):
        bonds = _read(mol2_dir / "naphthalene.mol2")["bonds"]
        assert bonds.n_rows == 11
        pairs = set(zip(bonds["atomi"].tolist(), bonds["atomj"].tolist()))
        assert {(5, 0), (9, 2), (1, 6)} <= pairs
        assert "type" not in bonds
        assert bonds["bond_type"].tolist() == [4] * 11
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
        with pytest.raises(ValueError):
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


def _mol2_text(atoms: list[tuple[str, str]]) -> str:
    """A one-molecule MOL2 of ``(name, type)`` atoms on a line, no bonds."""
    rows = "".join(
        f"{i + 1} {name} {float(i)} 0.0 0.0 {atype} 1 MOL 0.0\n"
        for i, (name, atype) in enumerate(atoms)
    )
    return (
        f"@<TRIPOS>MOLECULE\nM\n{len(atoms)} 0 0 0 0\nSMALL\nNO_CHARGES\n"
        f"@<TRIPOS>ATOM\n{rows}"
    )


def _elements(frame: mp.Frame) -> list[str | None]:
    atoms = frame["atoms"]
    valid = atoms.validity("element")
    values = list(atoms["element"])
    if valid is None:
        return values
    return [v if ok else None for v, ok in zip(values, valid)]


class TestMol2Elements:
    """MOL2 has no element field: the reader derives ``element`` from a SYBYL
    type, else from the atom name, and leaves an unspelled row null."""

    def test_gaff_typed_elements_come_from_the_names(self, tmp_path):
        path = tmp_path / "gaff.mol2"
        path.write_text(
            _mol2_text([("C1", "c3"), ("C2", "ca"), ("H3", "hc"), ("Br12", "br"), ("O5", "oh")]),
            encoding="utf-8",
        )
        assert _elements(_read(path)) == ["C", "C", "H", "Br", "O"]

    def test_sybyl_typed_elements_come_from_the_types(self, tmp_path):
        path = tmp_path / "sybyl.mol2"
        path.write_text(
            _mol2_text([("CA", "C.3"), ("X", "C.ar"), ("OXT", "O.3"), ("HO", "H"), ("BR", "Br")]),
            encoding="utf-8",
        )
        assert _elements(_read(path)) == ["C", "C", "O", "H", "Br"]

    def test_ambiguous_rows_are_null_not_guessed(self, tmp_path):
        path = tmp_path / "amber.mol2"
        path.write_text(_mol2_text([("CA", "CT"), ("C2", "c3")]), encoding="utf-8")
        assert _elements(_read(path)) == [None, "C"]

    def test_fixture_elements(self, mol2_dir):
        assert _elements(_read(mol2_dir / "ethane.mol2")) == ["C", "C"] + ["H"] * 6
        assert _elements(_read(mol2_dir / "li.mol2")) == ["Li"]

    def test_round_trip_keeps_elements(self, mol2_dir, tmp_path):
        frame = _read(mol2_dir / "ethane.mol2")
        out = tmp_path / "out.mol2"
        mp.io.write_mol2(out, frame)
        assert _elements(_read(out)) == _elements(frame)



def test_force_field_bond_type_names_do_not_replace_the_bond_order(tmp_path: Path) -> None:
    """A typed frame carries GAFF bond type names (``ca-ca``) in ``bonds.type``.
    They say nothing about the order, so the BOND records still get the SYBYL
    order from ``bond_type`` / ``bond_number`` and read back aromatic/double."""
    text = (
        "@<TRIPOS>MOLECULE\nX\n4 3 1 0 0\nSMALL\nNO_CHARGES\n\n@<TRIPOS>ATOM\n"
        "1 C1 0.0 0.0 0.0 ca 1 X\n2 C2 1.4 0.0 0.0 ca 1 X\n"
        "3 C3 2.8 0.0 0.0 c 1 X\n4 O1 4.0 0.0 0.0 o 1 X\n"
        "@<TRIPOS>BOND\n1 1 2 ar\n2 2 3 1\n3 3 4 2\n"
    )
    source = tmp_path / "in.mol2"
    source.write_text(text)
    frame = mp.io.read_mol2(source)
    frame["bonds"]["type"] = np.array(["ca-ca", "ca-c", "c-o"])
    out = tmp_path / "out.mol2"
    mp.io.write_mol2(out, frame)
    rows = out.read_text().split("@<TRIPOS>BOND\n", 1)[1].split()
    assert rows[3::4] == ["ar", "1", "2"]
    back = mp.io.read_mol2(out)
    assert list(back["bonds"]["bond_type"]) == [4, 1, 2]
