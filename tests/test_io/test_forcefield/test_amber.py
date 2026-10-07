"""``mp.io.read_amber_prmtop_system``: a prmtop's force field and structure frame."""

from __future__ import annotations

import numpy as np
import pytest

from molpy.ff.forcefield import AngleType, AtomType, BondType
from molpy.io import read_amber_prmtop_system


@pytest.fixture
def litfsi_prmtop(TEST_DATA_DIR):
    return TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop"


def test_prmtop_read_basic(litfsi_prmtop):
    ff, frame = read_amber_prmtop_system(litfsi_prmtop)
    assert frame is not None
    assert ff is not None
    assert "atoms" in frame
    assert "bonds" in frame
    assert "angles" in frame
    assert "dihedrals" in frame


def test_prmtop_read_pointers(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    assert frame.meta["n_atoms"] == 16
    assert frame["atoms"].nrows == 16
    assert "n_bonds" in frame.meta
    assert "n_angles" in frame.meta
    assert "n_dihedrals" in frame.meta


def test_prmtop_read_atom_names(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    names = frame["atoms"]["name"]
    assert len(names) == 16
    assert all(isinstance(n, str) for n in names)


def test_prmtop_read_charges(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    charges = np.asarray(frame["atoms"]["charge"], dtype=float)
    assert len(charges) == 16
    assert abs(charges[-1] - 1.0) < 1e-5  # Li+


def test_prmtop_read_atomic_numbers(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    atoms = frame["atoms"]
    # LiTFSI.prmtop carries %FLAG ATOMIC_NUMBER, so both columns must exist.
    z = np.asarray(atoms["atomic_number"])
    assert len(z) == 16
    assert all(z > 0)
    assert "element" in atoms


def test_prmtop_read_masses(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    masses = np.asarray(frame["atoms"]["mass"], dtype=float)
    assert len(masses) == 16
    assert all(masses > 0)


def test_prmtop_read_atom_types(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    types = frame["atoms"]["type"]
    assert types[0] == "f"
    assert types[1] == "c3"
    assert types[4] == "s6"
    assert types[7] == "ne"
    assert types[15] == "Li+"


def test_prmtop_read_bonds(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    bonds = frame["bonds"]
    for col in ("atomi", "atomj", "type", "type_id", "id"):
        assert col in bonds
    assert len(bonds["atomi"]) == 14


def test_prmtop_read_angles(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    angles = frame["angles"]
    for col in ("atomi", "atomj", "atomk", "type", "type_id", "id"):
        assert col in angles
    assert len(angles["atomi"]) == 25


def test_prmtop_read_dihedrals(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    dihedrals = frame["dihedrals"]
    for col in ("atomi", "atomj", "atomk", "atoml", "type", "id"):
        assert col in dihedrals
    # 27 prmtop rows (meta, from POINTERS); rows 11-16 are three two-term
    # torsions (`12 21 24 X 3` + `12 21 -24 X 4`), so 24 torsions.
    assert frame.meta["n_dihedrals"] == 27
    assert len(dihedrals["atomi"]) == 24
    assert all(0 <= i < 16 for i in dihedrals["atomi"])


def test_prmtop_read_residues(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    residues = np.asarray(frame["atoms"]["res_id"])
    assert len(residues) == 16
    assert all(isinstance(int(r), int) for r in residues)


def test_prmtop_forcefield_structure(litfsi_prmtop):
    ff, _ = read_amber_prmtop_system(litfsi_prmtop)
    assert ff.units == "real"
    assert hasattr(ff, "styles")
    assert len(ff.get_types(AtomType)) > 0
    assert len(ff.get_types(BondType)) > 0
    assert len(ff.get_types(AngleType)) > 0


def test_prmtop_nonexistent_file():
    with pytest.raises(ValueError, match="nonexistent"):
        read_amber_prmtop_system("/nonexistent/file.prmtop")


def test_bond_atom_indices_zero_based(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    bonds = frame["bonds"]
    n_atoms = frame.meta["n_atoms"]
    assert all(0 <= i < n_atoms for i in bonds["atomi"])
    assert all(0 <= j < n_atoms for j in bonds["atomj"])


def test_first_bond_atom_pair(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    bonds = frame["bonds"]
    pairs = set(
        zip(np.asarray(bonds["atomi"]).tolist(), np.asarray(bonds["atomj"]).tolist())
    )
    assert (11, 12) in pairs or (12, 11) in pairs


def test_charge_system_neutral(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    total = float(np.sum(np.asarray(frame["atoms"]["charge"], dtype=float)))
    assert abs(total) < 0.01


def test_charge_not_raw_units(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    charges = np.asarray(frame["atoms"]["charge"], dtype=float)
    assert all(abs(q) < 5.0 for q in charges)


def test_residue_atom_assignment_fsi(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    residues = np.asarray(frame["atoms"]["res_id"])
    assert all(residues[:15] == 0)
    assert residues[15] == 1


def test_residue_count(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    assert len(np.unique(np.asarray(frame["atoms"]["res_id"]))) == 2


def test_bond_residue_intra_fsi(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    res = np.asarray(frame["atoms"]["res_id"])
    bonds = frame["bonds"]
    for i, j in zip(bonds["atomi"], bonds["atomj"]):
        assert res[int(i)] == res[int(j)] == 0


def test_angle_residue_intra_fsi(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    res = np.asarray(frame["atoms"]["res_id"])
    angles = frame["angles"]
    for i, j, k in zip(angles["atomi"], angles["atomj"], angles["atomk"]):
        assert res[int(i)] == res[int(j)] == res[int(k)] == 0


def test_title_preserved_in_typed_meta(litfsi_prmtop):
    _, frame = read_amber_prmtop_system(litfsi_prmtop)
    assert frame.meta["title"] == "TFSI"


def test_missing_pointers_raises(tmp_path):
    bad = tmp_path / "empty.prmtop"
    bad.write_text("%VERSION 1\n%FLAG TITLE\n%FORMAT(20a4)\nx\n")
    with pytest.raises(ValueError, match="POINTERS"):
        read_amber_prmtop_system(bad)
