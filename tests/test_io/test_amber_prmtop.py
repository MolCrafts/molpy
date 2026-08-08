"""Tests for AMBER prmtop reading (structure + FF + molrs table helpers)."""

from __future__ import annotations

import math

import numpy as np
import pytest

import molpy.io as molpy_io
import molrs.io
from molpy import AngleType, AtomType, BondType
from molpy.io import read_amber
from molpy.io.forcefield.amber import CHARGE_CONVERSION_FACTOR, AmberPrmtopReader
from molrs import Frame

# LiTFSI POINTERS (first 31 integers in the fixture)
_LITFSI_POINTERS_LINES = [
    "      16       6       0      14       0      25       0      27       0       0",
    "      65       2      14      25      27       7      12       4       7       0",
    "       0       0       0       0       0       0       0       0      15       0",
    "       0",
]


@pytest.fixture
def litfsi_prmtop(TEST_DATA_DIR):
    return TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop"


@pytest.fixture
def litfsi_inpcrd(TEST_DATA_DIR):
    return TEST_DATA_DIR / "inpcrd" / "LiTFSI.inpcrd"


def test_read_amber_is_the_only_combined_amber_entry_point():
    assert molpy_io.read_amber is read_amber
    assert "read_amber_prmtop" not in molpy_io.__all__
    assert not hasattr(molpy_io, "read_amber_prmtop")


def test_prmtop_file_exists(litfsi_prmtop):
    assert litfsi_prmtop.exists()


def test_prmtop_reader_initialization(litfsi_prmtop):
    reader = AmberPrmtopReader(litfsi_prmtop)
    assert reader.file == litfsi_prmtop


def test_prmtop_read_basic(litfsi_prmtop):
    frame, ff = AmberPrmtopReader(litfsi_prmtop).read()
    assert frame is not None
    assert ff is not None
    assert "atoms" in frame
    assert "bonds" in frame
    assert "angles" in frame
    assert "dihedrals" in frame


def test_prmtop_read_into_caller_frame(litfsi_prmtop):
    dest = Frame()
    frame, ff = AmberPrmtopReader(litfsi_prmtop).read(dest)
    assert frame is dest
    assert dest["atoms"].nrows == 16
    assert ff is not None


def test_prmtop_read_pointers(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert frame.meta["n_atoms"].value == 16
    assert frame["atoms"].nrows == 16
    assert "n_bonds" in frame.meta
    assert "n_angles" in frame.meta
    assert "n_dihedrals" in frame.meta


def test_prmtop_read_atom_names(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    names = frame["atoms"]["name"]
    assert len(names) == 16
    assert all(isinstance(n, str) for n in names)


def test_prmtop_read_charges(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    charges = np.asarray(frame["atoms"]["charge"], dtype=float)
    assert len(charges) == 16
    assert abs(charges[-1] - 1.0) < 1e-5  # Li+


def test_prmtop_read_atomic_numbers(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    atoms = frame["atoms"]
    if "atomic_number" in atoms:
        z = np.asarray(atoms["atomic_number"])
        assert len(z) == 16
        assert all(z > 0)


def test_prmtop_read_masses(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    masses = np.asarray(frame["atoms"]["mass"], dtype=float)
    assert len(masses) == 16
    assert all(masses > 0)


def test_prmtop_read_atom_types(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    types = frame["atoms"]["type"]
    assert types[0] == "f"
    assert types[1] == "c3"
    assert types[4] == "s6"
    assert types[7] == "ne"
    assert types[15] == "Li+"


def test_prmtop_read_bonds(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    bonds = frame["bonds"]
    for col in ("atomi", "atomj", "type", "type_id", "id"):
        assert col in bonds
    assert len(bonds["atomi"]) == 14


def test_prmtop_read_angles(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    angles = frame["angles"]
    for col in ("atomi", "atomj", "atomk", "type", "type_id", "id"):
        assert col in angles
    assert len(angles["atomi"]) == 25


def test_prmtop_read_dihedrals(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    dihedrals = frame["dihedrals"]
    for col in ("atomi", "atomj", "atomk", "atoml", "type", "type_id", "id"):
        assert col in dihedrals
    n = frame.meta["n_dihedrals"].value
    assert len(dihedrals["atomi"]) == n
    assert all(0 <= i < 16 for i in dihedrals["atomi"])


def test_prmtop_read_residues(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    residues = np.asarray(frame["atoms"]["res_id"])
    assert len(residues) == 16
    assert all(isinstance(int(r), int) for r in residues)


def test_prmtop_forcefield_structure(litfsi_prmtop):
    _, ff = AmberPrmtopReader(litfsi_prmtop).read()
    assert ff.units == "real"
    assert hasattr(ff, "styles")
    assert len(ff.get_types(AtomType)) > 0
    assert len(ff.get_types(BondType)) > 0
    assert len(ff.get_types(AngleType)) > 0


def test_prmtop_charge_conversion_constant():
    assert CHARGE_CONVERSION_FACTOR == 18.2223


def test_prmtop_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        AmberPrmtopReader("/nonexistent/file.prmtop").read()


def test_read_amber_helper_reads_prmtop_and_inpcrd(litfsi_prmtop, litfsi_inpcrd):
    frame, ff = read_amber(litfsi_prmtop, litfsi_inpcrd)
    assert frame["atoms"].nrows == 16
    assert ff is not None
    assert "x" in frame["atoms"]
    assert "y" in frame["atoms"]
    assert "z" in frame["atoms"]


# ---------------------------------------------------------------------------
# molrs table helpers (not AmberPrmtopReader methods)
# ---------------------------------------------------------------------------


def test_prmtop_parse_pointers_raw_fields():
    meta = molrs.io.prmtop_parse_pointers(_LITFSI_POINTERS_LINES)
    assert meta["NATOM"] == 16
    assert meta["NTYPES"] == 6
    assert meta["MBONA"] == 14
    assert meta["MTHETA"] == 25
    assert meta["MPHIA"] == 27
    assert meta["NUMBND"] == 7
    assert meta["NUMANG"] == 12
    assert meta["NPTRA"] == 4
    assert meta["NATYP"] == 7
    assert meta["IFBOX"] == 0
    assert meta.get("NCOPY", 0) == 0


def test_prmtop_parse_pointers_derived_counts():
    meta = molrs.io.prmtop_parse_pointers(_LITFSI_POINTERS_LINES)
    assert meta["n_atoms"] == 16
    assert meta["n_bonds"] == 14
    assert meta["n_angles"] == 25
    assert meta["n_dihedrals"] == 27
    assert meta["n_atomtypes"] == 7
    assert meta["n_bondtypes"] == 7
    assert meta["n_angletypes"] == 12
    assert meta["n_dihedraltypes"] == 4


def test_prmtop_parse_pointers_30_values_graceful():
    meta = molrs.io.prmtop_parse_pointers(_LITFSI_POINTERS_LINES[:3])
    assert meta["n_atoms"] == 16
    assert meta["IFBOX"] == 0
    assert meta["NMXRS"] == 15
    assert "NUMEXTRA" not in meta
    assert "NCOPY" not in meta


def test_prmtop_parse_a4_names_chunking():
    line = "F   C   F1  F2  S   O   O3  N   "
    names = molrs.io.prmtop_parse_a4_names([line])
    assert names == ["F", "C", "F1", "F2", "S", "O", "O3", "N"]


def test_prmtop_parse_a4_names_strip():
    names = molrs.io.prmtop_parse_a4_names(["CA  CB  CG  "])
    assert names == ["CA", "CB", "CG"]


def test_prmtop_parse_a4_names_multiline():
    line = "F   C   F1  F2  S   O   O3  N   S1  O1  O2  C1  F4  F5  F3  LI  "
    names = molrs.io.prmtop_parse_a4_names([line])
    assert len(names) == 16
    assert names[0] == "F"
    assert names[-1] == "LI"


def test_prmtop_decode_bond_params_index_encoding():
    result = molrs.io.prmtop_decode_bond_params(
        [33, 36, 1], [359.0, 493.0], [1.3497, 1.466]
    )
    assert len(result) == 1
    bond_type, i, j, k, r0 = result[0]
    assert bond_type == 1
    assert i == 12
    assert j == 13
    assert abs(k - 359.0) < 1e-6
    assert abs(r0 - 1.3497) < 1e-6


def test_prmtop_decode_bond_params_sorted():
    result = molrs.io.prmtop_decode_bond_params([36, 33, 1], [359.0], [1.35])
    _, i, j, _, _ = result[0]
    assert i <= j


def test_prmtop_decode_bond_params_negative_raises():
    with pytest.raises(ValueError, match="negative bonded atom pointers"):
        molrs.io.prmtop_decode_bond_params([-3, 6, 1], [359.0], [1.35])


def test_bond_count_matches_pointers(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert frame.meta["n_bonds"].value == 14
    assert len(frame["bonds"]["atomi"]) == 14


def test_bond_atom_indices_zero_based(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    bonds = frame["bonds"]
    n_atoms = frame.meta["n_atoms"].value
    assert all(0 <= i < n_atoms for i in bonds["atomi"])
    assert all(0 <= j < n_atoms for j in bonds["atomj"])


def test_first_bond_atom_pair(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    bonds = frame["bonds"]
    pairs = set(
        zip(np.asarray(bonds["atomi"]).tolist(), np.asarray(bonds["atomj"]).tolist())
    )
    assert (11, 12) in pairs or (12, 11) in pairs


def test_prmtop_decode_angle_params_returns_degrees():
    angles = molrs.io.prmtop_decode_angle_params(
        [39, 33, 42, 1],
        [71.0],
        [1.87378629],
    )
    assert len(angles) == 1
    _, _i, _j, _k, _f, theta = angles[0]
    assert 0 < theta < 180
    assert abs(theta - math.degrees(1.87378629)) < 1e-4


def test_angle_equil_from_litfsi_tables(litfsi_prmtop):
    sec = molrs.io.read_amber_prmtop_sections(str(litfsi_prmtop))
    ptrs = [int(x) for line in sec["ANGLES_WITHOUT_HYDROGEN"] for x in line.split()]
    ptrs += [
        int(x) for line in sec.get("ANGLES_INC_HYDROGEN", []) for x in line.split()
    ]
    k = [float(x) for line in sec["ANGLE_FORCE_CONSTANT"] for x in line.split()]
    teq = [float(x) for line in sec["ANGLE_EQUIL_VALUE"] for x in line.split()]
    angles = molrs.io.prmtop_decode_angle_params(ptrs, k, teq)
    type1 = [a for a in angles if a[0] == 1]
    assert type1
    expected = math.degrees(1.87378629)
    assert any(abs(a[5] - expected) < 0.01 for a in type1)


def test_angle_count_matches_pointers(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert frame.meta["n_angles"].value == 25
    assert len(frame["angles"]["atomi"]) == 25


def test_prmtop_decode_dihedral_negative_k():
    dih = molrs.io.prmtop_decode_dihedral_params(
        [12, 21, -24, 27, 1],
        [0.5],
        [3.14159265],
        [3.0],
    )
    assert len(dih) == 1
    _, _i, _j, k, l, *_ = dih[0]
    assert k == 9
    assert l == 10


def test_prmtop_decode_dihedral_negative_l_improper():
    dih = molrs.io.prmtop_decode_dihedral_params(
        [0, 6, 12, -18, 1],
        [0.5],
        [0.0],
        [2.0],
    )
    assert dih[0][4] == 7


def test_prmtop_decode_dihedral_periodicity_int():
    dih = molrs.io.prmtop_decode_dihedral_params(
        [0, 6, 12, 18, 1],
        [0.5],
        [0.0],
        [3.0],
    )
    assert isinstance(dih[0][7], int)
    assert dih[0][7] == 3


def test_dihedral_count_matches_pointers(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert frame.meta["n_dihedrals"].value == 27
    assert len(frame["dihedrals"]["atomi"]) == 27


def test_dihedral_indices_in_range(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    n_atoms = frame.meta["n_atoms"].value
    for key in ("atomi", "atomj", "atomk", "atoml"):
        assert all(0 <= v < n_atoms for v in frame["dihedrals"][key])


def test_charge_conversion_factor_value():
    assert CHARGE_CONVERSION_FACTOR == 18.2223


def test_charge_li_equals_one(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert abs(float(np.asarray(frame["atoms"]["charge"])[-1]) - 1.0) < 1e-5


def test_charge_system_neutral(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    total = float(np.sum(np.asarray(frame["atoms"]["charge"], dtype=float)))
    assert abs(total) < 0.01


def test_charge_not_raw_units(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    charges = np.asarray(frame["atoms"]["charge"], dtype=float)
    assert all(abs(q) < 5.0 for q in charges)


def test_nonbond_sigma_epsilon_formula():
    acoef = 2.06132451e5
    bcoef = 2.24268406e2
    r_min = (2 * acoef / bcoef) ** (1 / 6)
    eps = 0.25 * bcoef**2 / acoef
    sigma = 2 ** (-1 / 6) * r_min
    assert abs(sigma - 3.117) < 0.01
    assert abs(eps - 0.061) < 0.002


def test_prmtop_decode_nonbond_zero_acoef():
    result = molrs.io.prmtop_decode_nonbond_params(1, 1, [1], [1], [0.0], [0.0], [], [])
    assert len(result) == 1
    _, sigma, epsilon = result[0]
    assert epsilon == 0.0
    assert sigma == 1.0


def test_nonbond_li_sigma_epsilon(litfsi_prmtop):
    sec = molrs.io.read_amber_prmtop_sections(str(litfsi_prmtop))
    meta = molrs.io.prmtop_parse_pointers(sec["POINTERS"])
    params = molrs.io.prmtop_decode_nonbond_params(
        int(meta["NATOM"]),
        int(meta["NTYPES"]),
        [int(x) for line in sec["ATOM_TYPE_INDEX"] for x in line.split()],
        [int(x) for line in sec["NONBONDED_PARM_INDEX"] for x in line.split()],
        [float(x) for line in sec["LENNARD_JONES_ACOEF"] for x in line.split()],
        [float(x) for line in sec["LENNARD_JONES_BCOEF"] for x in line.split()],
        [float(x) for line in sec.get("HBOND_ACOEF", []) for x in line.split()],
        [float(x) for line in sec.get("HBOND_BCOEF", []) for x in line.split()],
    )
    li = [p for p in params if p[0] == 16]
    assert li
    _, sigma, epsilon = li[0]
    assert 1.5 < sigma < 2.2
    assert 0.01 < epsilon < 0.1


def test_nonbond_f_sigma_epsilon(litfsi_prmtop):
    sec = molrs.io.read_amber_prmtop_sections(str(litfsi_prmtop))
    meta = molrs.io.prmtop_parse_pointers(sec["POINTERS"])
    params = molrs.io.prmtop_decode_nonbond_params(
        int(meta["NATOM"]),
        int(meta["NTYPES"]),
        [int(x) for line in sec["ATOM_TYPE_INDEX"] for x in line.split()],
        [int(x) for line in sec["NONBONDED_PARM_INDEX"] for x in line.split()],
        [float(x) for line in sec["LENNARD_JONES_ACOEF"] for x in line.split()],
        [float(x) for line in sec["LENNARD_JONES_BCOEF"] for x in line.split()],
    )
    f = [p for p in params if p[0] == 1]
    assert f
    _, sigma, epsilon = f[0]
    assert abs(sigma - 3.117) < 0.05
    assert abs(epsilon - 0.061) < 0.005


def test_prmtop_decode_nonbond_10_12_raises():
    with pytest.raises(ValueError, match="10-12 interactions"):
        molrs.io.prmtop_decode_nonbond_params(
            1,
            1,
            [1],
            [1],
            [1.0],
            [1.0],
            [1000.0],
            [0.5],
        )


def test_residue_atom_assignment_fsi(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    residues = np.asarray(frame["atoms"]["res_id"])
    assert all(residues[:15] == 0)
    assert residues[15] == 1


def test_residue_count(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    assert len(np.unique(np.asarray(frame["atoms"]["res_id"]))) == 2


def test_bond_residue_intra_fsi(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    res = np.asarray(frame["atoms"]["res_id"])
    bonds = frame["bonds"]
    for i, j in zip(bonds["atomi"], bonds["atomj"]):
        assert res[int(i)] == res[int(j)] == 0


def test_angle_residue_intra_fsi(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    res = np.asarray(frame["atoms"]["res_id"])
    angles = frame["angles"]
    for i, j, k in zip(angles["atomi"], angles["atomj"], angles["atomk"]):
        assert res[int(i)] == res[int(j)] == res[int(k)] == 0


def test_title_preserved_in_typed_meta(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    if "title" in frame.meta:
        assert "TFSI" in str(frame.meta["title"].value) or str(
            frame.meta["title"].value
        )


def test_missing_pointers_raises_valueerror(tmp_path):
    bad = tmp_path / "empty.prmtop"
    bad.write_text("%VERSION 1\n%FLAG TITLE\n%FORMAT(20a4)\nx\n")
    with pytest.raises(ValueError, match="POINTERS"):
        AmberPrmtopReader(bad).read()


def test_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        AmberPrmtopReader("/no/such/file.prmtop").read()


def test_known_elements_become_an_atomic_number_column(litfsi_prmtop):
    frame, _ = AmberPrmtopReader(litfsi_prmtop).read()
    atoms = frame["atoms"]
    if "atomic_number" in atoms:
        assert "element" in atoms
        assert len(atoms["atomic_number"]) == 16


def test_bonds_with_h_table_from_sections(litfsi_prmtop):
    """BONDS_INC_HYDROGEN empty for LiTFSI; WITHOUT_H has 14."""
    sec = molrs.io.read_amber_prmtop_sections(str(litfsi_prmtop))
    inc = [int(x) for line in sec.get("BONDS_INC_HYDROGEN", []) for x in line.split()]
    wout = [
        int(x) for line in sec.get("BONDS_WITHOUT_HYDROGEN", []) for x in line.split()
    ]
    k = [float(x) for line in sec["BOND_FORCE_CONSTANT"] for x in line.split()]
    r0 = [float(x) for line in sec["BOND_EQUIL_VALUE"] for x in line.split()]
    assert molrs.io.prmtop_decode_bond_params(inc, k, r0) == []
    assert len(molrs.io.prmtop_decode_bond_params(wout, k, r0)) == 14
