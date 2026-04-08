"""Tests for AMBER prmtop file reading."""

import numpy as np
import pytest

from molpy.core.frame import Frame
from molpy.io import read_amber
from molpy.io.forcefield.amber import AmberPrmtopReader, CHARGE_CONVERSION_FACTOR


@pytest.fixture
def litfsi_prmtop(TEST_DATA_DIR):
    """Path to LiTFSI test prmtop file."""
    return TEST_DATA_DIR / "prmtop" / "LiTFSI.prmtop"


@pytest.fixture
def litfsi_inpcrd(TEST_DATA_DIR):
    """Path to LiTFSI test inpcrd file."""
    return TEST_DATA_DIR / "inpcrd" / "LiTFSI.inpcrd"


def test_prmtop_file_exists(litfsi_prmtop):
    """Test that the test prmtop file exists."""
    assert litfsi_prmtop.exists(), f"Test file not found: {litfsi_prmtop}"


def test_prmtop_reader_initialization(litfsi_prmtop):
    """Test AmberPrmtopReader initialization."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    assert reader.file == litfsi_prmtop
    assert reader.raw_data == {}
    assert reader.meta == {}


def test_prmtop_sanitizer():
    """Test the sanitizer static method."""
    assert AmberPrmtopReader.sanitizer("  test  \n") == "test"
    assert AmberPrmtopReader.sanitizer("\t  data  ") == "data"
    assert AmberPrmtopReader.sanitizer("nowhitespace") == "nowhitespace"


def test_prmtop_read_section_int():
    """Test read_section with integer conversion."""
    lines = ["1 2 3", "4 5 6"]
    result = AmberPrmtopReader.read_section(lines, int)
    assert result == [1, 2, 3, 4, 5, 6]


def test_prmtop_read_section_float():
    """Test read_section with float conversion."""
    lines = ["1.0 2.5", "3.7 4.2"]
    result = AmberPrmtopReader.read_section(lines, float)
    assert result == [1.0, 2.5, 3.7, 4.2]


def test_prmtop_read_section_str():
    """Test read_section with string conversion."""
    lines = ["ABC DEF", "GHI"]
    result = AmberPrmtopReader.read_section(lines, str)
    assert result == ["ABC", "DEF", "GHI"]


def test_prmtop_read_basic(litfsi_prmtop):
    """Test basic reading of prmtop file."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    # Check that frame and forcefield were returned
    assert frame is not None
    assert ff is not None

    # Check that basic blocks exist
    assert "atoms" in frame
    assert "bonds" in frame
    assert "angles" in frame
    assert "dihedrals" in frame


def test_prmtop_read_pointers(litfsi_prmtop):
    """Test reading POINTERS section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    # LiTFSI has 16 atoms (15 TFSI + 1 Li)
    assert frame.metadata["n_atoms"] == 16
    assert frame["atoms"].nrows == 16

    # Check that meta contains expected fields
    assert "n_bonds" in frame.metadata
    assert "n_angles" in frame.metadata
    assert "n_dihedrals" in frame.metadata
    assert "n_atomtypes" in frame.metadata


def test_prmtop_read_atom_names(litfsi_prmtop):
    """Test reading ATOM_NAME section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "name" in atoms

    # Check some expected atom names
    names = atoms["name"]
    assert len(names) == 16
    # First few atoms should be F, C, F1, F2, S, ...
    assert names[0] == "F"
    assert names[1] == "C"
    # Last atom should be LI
    assert names[-1] == "LI"


def test_prmtop_read_charges(litfsi_prmtop):
    """Test reading and converting CHARGE section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "charge" in atoms

    charges = atoms["charge"]
    assert len(charges) == 16

    # Charges should be converted by dividing by CHARGE_CONVERSION_FACTOR
    # Last atom (Li+) should have charge close to +1
    assert np.isclose(charges[-1], 1.0, atol=0.01)

    # Check that charges sum to approximately 0 (neutral system)
    # Note: LiTFSI is Li+ with TFSI-, so should sum to 0
    total_charge = np.sum(charges)
    assert np.isclose(total_charge, 0.0, atol=0.01)


def test_prmtop_read_atomic_numbers(litfsi_prmtop):
    """Test reading ATOMIC_NUMBER section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "atomic_number" in atoms

    atomic_numbers = atoms["atomic_number"]
    assert len(atomic_numbers) == 16

    # Li has atomic number 3
    assert atomic_numbers[-1] == 3
    # F has atomic number 9
    assert atomic_numbers[0] == 9


def test_prmtop_read_masses(litfsi_prmtop):
    """Test reading MASS section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "mass" in atoms

    masses = atoms["mass"]
    assert len(masses) == 16

    # Li mass ~6.94
    assert np.isclose(masses[-1], 6.94, atol=0.1)
    # F mass ~19.0
    assert np.isclose(masses[0], 19.0, atol=0.5)


def test_prmtop_read_atom_types(litfsi_prmtop):
    """Test reading AMBER_ATOM_TYPE section."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "type" in atoms

    types = atoms["type"]
    assert len(types) == 16
    # All should be strings
    assert all(isinstance(t, str) for t in types)


def test_prmtop_read_bonds(litfsi_prmtop):
    """Test reading bond information."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    bonds = frame["bonds"]
    assert "atomi" in bonds
    assert "atomj" in bonds
    assert "type" in bonds
    assert "type_id" in bonds
    assert "id" in bonds

    n_bonds = frame.metadata["n_bonds"]
    assert len(bonds["atomi"]) == n_bonds
    assert len(bonds["atomj"]) == n_bonds

    # Bond indices should be valid (0-indexed, less than n_atoms)
    assert all(0 <= i < 16 for i in bonds["atomi"])
    assert all(0 <= j < 16 for j in bonds["atomj"])


def test_prmtop_read_angles(litfsi_prmtop):
    """Test reading angle information."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    angles = frame["angles"]
    assert "atomi" in angles
    assert "atomj" in angles
    assert "atomk" in angles
    assert "type" in angles
    assert "type_id" in angles
    assert "id" in angles

    n_angles = frame.metadata["n_angles"]
    assert len(angles["atomi"]) == n_angles
    assert len(angles["atomj"]) == n_angles
    assert len(angles["atomk"]) == n_angles

    # Angle indices should be valid
    assert all(0 <= i < 16 for i in angles["atomi"])
    assert all(0 <= j < 16 for j in angles["atomj"])
    assert all(0 <= k < 16 for k in angles["atomk"])


def test_prmtop_read_dihedrals(litfsi_prmtop):
    """Test reading dihedral information."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    dihedrals = frame["dihedrals"]
    assert "atomi" in dihedrals
    assert "atomj" in dihedrals
    assert "atomk" in dihedrals
    assert "atoml" in dihedrals
    assert "type" in dihedrals
    assert "type_id" in dihedrals
    assert "id" in dihedrals

    n_dihedrals = frame.metadata["n_dihedrals"]
    assert len(dihedrals["atomi"]) == n_dihedrals
    assert len(dihedrals["atomj"]) == n_dihedrals
    assert len(dihedrals["atomk"]) == n_dihedrals
    assert len(dihedrals["atoml"]) == n_dihedrals

    # Dihedral indices should be valid
    assert all(0 <= i < 16 for i in dihedrals["atomi"])
    assert all(0 <= j < 16 for j in dihedrals["atomj"])
    assert all(0 <= k < 16 for k in dihedrals["atomk"])
    assert all(0 <= l < 16 for l in dihedrals["atoml"])


def test_prmtop_read_residues(litfsi_prmtop):
    """Test reading residue information."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    assert "residue" in atoms

    residues = atoms["residue"]
    assert len(residues) == 16

    # Should have residue assignments for all atoms
    assert all(isinstance(r, (int, np.integer)) for r in residues)


def test_prmtop_forcefield_structure(litfsi_prmtop):
    """Test that forcefield has correct structure."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    # Check forcefield units
    assert ff.units == "real"

    # Check that forcefield has styles attribute
    assert hasattr(ff, "styles")
    assert ff.styles is not None

    # Check that we can get atom types
    atomtypes = ff.get_atomtypes()
    assert len(atomtypes) > 0

    # Check that we can get bond types
    bondtypes = ff.get_bondtypes()
    assert len(bondtypes) > 0

    # Check that we can get angle types
    angletypes = ff.get_angletypes()
    assert len(angletypes) > 0


def test_prmtop_charge_conversion_constant():
    """Test that CHARGE_CONVERSION_FACTOR is defined correctly."""
    assert CHARGE_CONVERSION_FACTOR == 18.2223


def test_prmtop_nonexistent_file():
    """Test error handling for nonexistent file."""
    reader = AmberPrmtopReader("/nonexistent/file.prmtop")
    frame = Frame()

    with pytest.raises(FileNotFoundError):
        reader.read(frame)


def test_prmtop_get_bond_with_H(litfsi_prmtop):
    """Test get_bond_with_H method."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    reader.read(frame)

    # Should return list of tuples
    bonds_with_h = reader.get_bond_with_H()
    assert isinstance(bonds_with_h, list)

    # Each tuple should have 5 elements: (type, i, j, force_constant, equil_length)
    for bond in bonds_with_h:
        assert len(bond) == 5
        assert isinstance(bond[0], int)  # type
        assert isinstance(bond[1], int)  # i
        assert isinstance(bond[2], int)  # j
        assert isinstance(bond[3], float)  # force constant
        assert isinstance(bond[4], float)  # equilibrium length


def test_prmtop_get_bond_without_H(litfsi_prmtop):
    """Test get_bond_without_H method."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    reader.read(frame)

    # Should return list of tuples
    bonds_without_h = reader.get_bond_without_H()
    assert isinstance(bonds_without_h, list)

    # Each tuple should have 5 elements
    for bond in bonds_without_h:
        assert len(bond) == 5
        assert isinstance(bond[0], int)  # type
        assert isinstance(bond[1], int)  # i
        assert isinstance(bond[2], int)  # j
        assert isinstance(bond[3], float)  # force constant
        assert isinstance(bond[4], float)  # equilibrium length


def test_prmtop_parse_angle_params(litfsi_prmtop):
    """Test parse_angle_params method."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    reader.read(frame)

    # Should return list of tuples
    angles = reader.parse_angle_params()
    assert isinstance(angles, list)

    # Each tuple should have 6 elements: (type, i, j, k, force_constant, equil_angle)
    for angle in angles:
        assert len(angle) == 6
        assert isinstance(angle[0], int)  # type
        assert isinstance(angle[1], int)  # i
        assert isinstance(angle[2], int)  # j
        assert isinstance(angle[3], int)  # k
        assert isinstance(angle[4], float)  # force constant
        assert isinstance(angle[5], float)  # equilibrium angle (in degrees)
        # Angle should be in reasonable range (0-180 degrees)
        assert 0 <= angle[5] <= 180


def test_prmtop_parse_dihedral_params(litfsi_prmtop):
    """Test parse_dihedral_params method."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    reader.read(frame)

    # Should return list of tuples
    dihedrals = reader.parse_dihedral_params()
    assert isinstance(dihedrals, list)

    # Each tuple should have 8 elements: (type, i, j, k, l, force_constant, phase, periodicity)
    for dihedral in dihedrals:
        assert len(dihedral) == 8
        assert isinstance(dihedral[0], int)  # type
        assert isinstance(dihedral[1], int)  # i
        assert isinstance(dihedral[2], int)  # j
        assert isinstance(dihedral[3], int)  # k
        assert isinstance(dihedral[4], int)  # l
        assert isinstance(dihedral[5], float)  # force constant
        assert isinstance(dihedral[6], float)  # phase
        assert isinstance(dihedral[7], int)  # periodicity


def test_prmtop_parse_nonbond_params(litfsi_prmtop):
    """Test parse_nonbond_params method."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, ff = reader.read(frame)

    atoms = frame["atoms"]
    nonbond_params = reader.parse_nonbond_params(atoms)

    assert isinstance(nonbond_params, list)
    assert len(nonbond_params) == 16  # One per atom

    # Each tuple should have 3 elements: (atom_index, sigma, epsilon)
    for param in nonbond_params:
        assert len(param) == 3
        assert isinstance(param[0], int)  # atom index (1-based)
        assert isinstance(param[1], float)  # sigma
        assert isinstance(param[2], float)  # epsilon
        # Sigma and epsilon should be non-negative
        assert param[1] >= 0
        assert param[2] >= 0


def test_read_amber_helper_reads_prmtop_and_inpcrd(litfsi_prmtop, litfsi_inpcrd):
    """read_amber helper should return frame+ff with coordinates loaded."""
    frame, ff = read_amber(litfsi_prmtop, litfsi_inpcrd)

    assert frame["atoms"].nrows == 16
    assert ff is not None
    assert "x" in frame["atoms"]
    assert "y" in frame["atoms"]
    assert "z" in frame["atoms"]


# ---------------------------------------------------------------------------
# LiTFSI ground-truth POINTERS values (from the first 32 integers in the file)
# ---------------------------------------------------------------------------
_LITFSI_POINTERS_LINES = [
    "      16       6       0      14       0      25       0      27       0       0",
    "      65       2      14      25      27       7      12       4       7       0",
    "       0       0       0       0       0       0       0       0      15       0",
    "       0",
]


# ---------------------------------------------------------------------------
# _read_pointers unit tests
# ---------------------------------------------------------------------------


def test_read_pointers_raw_fields():
    """All 32 POINTERS fields parse to correct integer values."""
    reader = AmberPrmtopReader("dummy")
    meta = reader._read_pointers(_LITFSI_POINTERS_LINES)
    assert meta["NATOM"] == 16
    assert meta["NTYPES"] == 6
    assert meta["NBONH"] == 0
    assert meta["MBONA"] == 14
    assert meta["NTHETH"] == 0
    assert meta["MTHETA"] == 25
    assert meta["NPHIH"] == 0
    assert meta["MPHIA"] == 27
    assert meta["NNB"] == 65
    assert meta["NRES"] == 2
    assert meta["NBONA"] == 14
    assert meta["NTHETA"] == 25
    assert meta["NPHIA"] == 27
    assert meta["NUMBND"] == 7
    assert meta["NUMANG"] == 12
    assert meta["NPTRA"] == 4
    assert meta["NATYP"] == 7
    assert meta["IFBOX"] == 0
    assert meta["NMXRS"] == 15
    assert meta["IFCAP"] == 0
    assert meta["NUMEXTRA"] == 0
    # LiTFSI.prmtop has 31 POINTERS values; NCOPY (index 31) may be absent
    assert meta.get("NCOPY", 0) == 0


def test_read_pointers_derived_counts():
    """Derived counts (n_bonds, n_angles, n_dihedrals) use NBONH+MBONA etc."""
    reader = AmberPrmtopReader("dummy")
    meta = reader._read_pointers(_LITFSI_POINTERS_LINES)
    assert meta["n_atoms"] == 16  # NATOM
    assert meta["n_bonds"] == 14  # NBONH(0) + MBONA(14)
    assert meta["n_angles"] == 25  # NTHETH(0) + MTHETA(25)
    assert meta["n_dihedrals"] == 27  # NPHIH(0) + MPHIA(27)
    assert meta["n_atomtypes"] == 7  # NATYP
    assert meta["n_bondtypes"] == 7  # NUMBND
    assert meta["n_angletypes"] == 12  # NUMANG
    assert meta["n_dihedraltypes"] == 4  # NPTRA


def test_read_pointers_30_values_graceful():
    """Old prmtop files with only 30 POINTERS values are handled gracefully."""
    reader = AmberPrmtopReader("dummy")
    lines_30 = _LITFSI_POINTERS_LINES[:3]  # 30 values (3 lines of 10)
    meta = reader._read_pointers(lines_30)
    assert meta["n_atoms"] == 16
    assert meta["IFBOX"] == 0
    assert meta["NMXRS"] == 15
    assert "NUMEXTRA" not in meta  # gracefully absent
    assert "NCOPY" not in meta


# ---------------------------------------------------------------------------
# _read_atom_name unit tests  (20a4 fixed-width format)
# ---------------------------------------------------------------------------


def test_read_atom_name_4char_chunking():
    """Names are extracted as 4-char fixed-width fields, not whitespace-split."""
    reader = AmberPrmtopReader("dummy")
    line = "F   C   F1  F2  S   O   O3  N   "
    names = reader._read_atom_name([line])
    assert names == ["F", "C", "F1", "F2", "S", "O", "O3", "N"]


def test_read_atom_name_trailing_spaces_stripped():
    """Trailing spaces within each 4-char slot are stripped."""
    reader = AmberPrmtopReader("dummy")
    names = reader._read_atom_name(["CA  CB  CG  "])
    assert names == ["CA", "CB", "CG"]
    assert all(not n.endswith(" ") for n in names)


def test_read_atom_name_multiline():
    """Names split across multiple lines are concatenated correctly."""
    reader = AmberPrmtopReader("dummy")
    line1 = "F   C   F1  F2  S   O   O3  N   S1  O1  O2  C1  F4  F5  F3  LI  "
    names = reader._read_atom_name([line1])
    expected = [
        "F",
        "C",
        "F1",
        "F2",
        "S",
        "O",
        "O3",
        "N",
        "S1",
        "O1",
        "O2",
        "C1",
        "F4",
        "F5",
        "F3",
        "LI",
    ]
    assert names == expected
    assert len(names) == 16


def test_amber_atom_type_4char(litfsi_prmtop):
    """AMBER_ATOM_TYPE is parsed with 4-char fixed-width (same as ATOM_NAME)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    types = frame["atoms"]["type"]
    # From the file: f c3 f f s6 o o ne sy o o c3 f f f Li+
    assert types[0] == "f"
    assert types[1] == "c3"
    assert types[4] == "s6"
    assert types[7] == "ne"
    assert types[15] == "Li+"


# ---------------------------------------------------------------------------
# Bond encoding tests  (raw // 3 + 1 → 1-based atom index)
# ---------------------------------------------------------------------------


def test_parse_bond_params_index_encoding():
    """Raw bond pointers decoded as raw//3+1 (1-based)."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "BOND_FORCE_CONSTANT": [359.0, 493.0],
        "BOND_EQUIL_VALUE": [1.3497, 1.466],
    }
    # raw=33 → 33//3+1=12;  raw=36 → 36//3+1=13;  type=1
    result = reader._parse_bond_params([33, 36, 1])
    assert len(result) == 1
    bond_type, i, j, k, r0 = result[0]
    assert bond_type == 1
    assert i == 12
    assert j == 13
    assert abs(k - 359.0) < 1e-6
    assert abs(r0 - 1.3497) < 1e-6


def test_parse_bond_params_sorted():
    """Atom indices in each bond are returned in ascending order."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "BOND_FORCE_CONSTANT": [359.0],
        "BOND_EQUIL_VALUE": [1.35],
    }
    # Provide j < i in raw to confirm sorting
    result = reader._parse_bond_params([36, 33, 1])
    _, i, j, _, _ = result[0]
    assert i <= j


def test_parse_bond_params_negative_raises():
    """Negative bond atom pointers raise an Exception."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "BOND_FORCE_CONSTANT": [359.0],
        "BOND_EQUIL_VALUE": [1.35],
    }
    with pytest.raises(Exception, match="negative bonded atom pointers"):
        reader._parse_bond_params([-3, 6, 1])


def test_bond_count_matches_pointers(litfsi_prmtop):
    """Number of bonds in frame == NBONH + MBONA from POINTERS."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    # NBONH=0, MBONA=14 → n_bonds=14
    assert frame.metadata["n_bonds"] == 14
    assert len(frame["bonds"]["atomi"]) == 14


def test_bond_atom_indices_zero_based(litfsi_prmtop):
    """Bond atom indices in frame are 0-based and within [0, n_atoms)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    bonds = frame["bonds"]
    n_atoms = frame.metadata["n_atoms"]
    assert all(0 <= i < n_atoms for i in bonds["atomi"])
    assert all(0 <= j < n_atoms for j in bonds["atomj"])


def test_first_bond_atom_pair(litfsi_prmtop):
    """First BONDS_WITHOUT_HYDROGEN triplet (33,36,1) decodes to atoms 11,12 (0-based)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    bonds = frame["bonds"]
    pairs = set(
        zip(np.asarray(bonds["atomi"]).tolist(), np.asarray(bonds["atomj"]).tolist())
    )
    # 33//3+1=12, 36//3+1=13 → 0-based: 11 and 12
    assert (11, 12) in pairs or (12, 11) in pairs


# ---------------------------------------------------------------------------
# Angle encoding tests
# ---------------------------------------------------------------------------


def test_parse_angle_params_returns_degrees():
    """Angle equilibrium values are converted from radians to degrees."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "ANGLE_FORCE_CONSTANT": [71.0],
        "ANGLE_EQUIL_VALUE": [1.87378629],  # radians
        "ANGLES_INC_HYDROGEN": [],
        "ANGLES_WITHOUT_HYDROGEN": [39, 33, 42, 1],
    }
    angles = reader.parse_angle_params()
    assert len(angles) == 1
    _, i, j, k, f, theta = angles[0]
    # Must be in degrees (~107.32), not radians (~1.87)
    assert 0 < theta < 180, f"Angle {theta} not in degrees (0-180)"
    import math

    assert abs(theta - math.degrees(1.87378629)) < 1e-4


def test_angle_equil_value_from_litfsi(litfsi_prmtop):
    """First ANGLE_EQUIL_VALUE (1.87378629 rad) appears as ~107.32° in parsed output."""
    import math

    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    angles = reader.parse_angle_params()
    # Find an angle of type 1 (first ANGLE_EQUIL_VALUE entry)
    type1_angles = [a for a in angles if a[0] == 1]
    assert type1_angles, "Expected at least one angle of type 1"
    expected_deg = math.degrees(1.87378629)
    assert any(abs(a[5] - expected_deg) < 0.01 for a in type1_angles)


def test_angle_count_matches_pointers(litfsi_prmtop):
    """Number of angles == NTHETH + MTHETA from POINTERS."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    assert frame.metadata["n_angles"] == 25
    assert len(frame["angles"]["atomi"]) == 25


# ---------------------------------------------------------------------------
# Dihedral encoding tests  (negative k = end-group; negative l = improper)
# ---------------------------------------------------------------------------


def test_dihedral_negative_k_end_group():
    """Negative raw k (3rd atom) signals end-group suppression; abs() applied."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "DIHEDRAL_FORCE_CONSTANT": [0.5],
        "DIHEDRAL_PHASE": [3.14159265],
        "DIHEDRAL_PERIODICITY": [3.0],
        "DIHEDRALS_INC_HYDROGEN": [],
        # (12, 21, -24, 27, 1): k=-24 → abs(-24)//3+1 = 9
        "DIHEDRALS_WITHOUT_HYDROGEN": [12, 21, -24, 27, 1],
    }
    dihedrals = reader.parse_dihedral_params()
    assert len(dihedrals) == 1
    _, i, j, k, l, *_ = dihedrals[0]
    assert k == 9, f"Expected abs(-24)//3+1=9, got {k}"
    assert l == 10, f"Expected 27//3+1=10, got {l}"


def test_dihedral_negative_l_improper():
    """Negative raw l (4th atom) signals improper torsion; abs() applied."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "DIHEDRAL_FORCE_CONSTANT": [0.5],
        "DIHEDRAL_PHASE": [0.0],
        "DIHEDRAL_PERIODICITY": [2.0],
        "DIHEDRALS_INC_HYDROGEN": [],
        # l=-18 → abs(-18)//3+1 = 7
        "DIHEDRALS_WITHOUT_HYDROGEN": [0, 6, 12, -18, 1],
    }
    dihedrals = reader.parse_dihedral_params()
    assert len(dihedrals) == 1
    _, i, j, k, l, *_ = dihedrals[0]
    assert l == 7, f"Expected abs(-18)//3+1=7, got {l}"


def test_dihedral_periodicity_is_int():
    """Dihedral periodicity is stored as float but returned as int."""
    reader = AmberPrmtopReader("dummy")
    reader.raw_data = {
        "DIHEDRAL_FORCE_CONSTANT": [0.5],
        "DIHEDRAL_PHASE": [0.0],
        "DIHEDRAL_PERIODICITY": [3.0],
        "DIHEDRALS_INC_HYDROGEN": [],
        "DIHEDRALS_WITHOUT_HYDROGEN": [0, 6, 12, 18, 1],
    }
    dihedrals = reader.parse_dihedral_params()
    periodicity = dihedrals[0][7]
    assert isinstance(periodicity, int)
    assert periodicity == 3


def test_dihedral_count_matches_pointers(litfsi_prmtop):
    """Number of dihedrals == NPHIH + MPHIA from POINTERS."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    assert frame.metadata["n_dihedrals"] == 27
    assert len(frame["dihedrals"]["atomi"]) == 27


def test_dihedral_negative_atoms_in_litfsi(litfsi_prmtop):
    """LiTFSI dihedrals with negative k atoms are decoded with abs() (no crash)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    n_atoms = frame.metadata["n_atoms"]
    for key in ("atomi", "atomj", "atomk", "atoml"):
        assert all(0 <= v < n_atoms for v in frame["dihedrals"][key]), (
            f"Dihedral {key} out of range"
        )


# ---------------------------------------------------------------------------
# Charge conversion tests
# ---------------------------------------------------------------------------


def test_charge_conversion_factor_value():
    """CHARGE_CONVERSION_FACTOR equals 18.2223 (sqrt of 332.0636 kcal·Å/mol/e²)."""
    assert CHARGE_CONVERSION_FACTOR == 18.2223


def test_charge_li_equals_one(litfsi_prmtop):
    """Li+ stored charge (18.2223) converts to exactly +1.0 e."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    li_charge = np.asarray(frame["atoms"]["charge"])[-1]
    assert abs(li_charge - 1.0) < 1e-5


def test_charge_system_neutral(litfsi_prmtop):
    """LiTFSI is charge-neutral; sum of converted charges ≈ 0 (within float precision)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    total = np.sum(np.asarray(frame["atoms"]["charge"], dtype=float))
    assert abs(total) < 0.01  # floating-point rounding in file; true sum is 0


def test_charge_not_raw_units(litfsi_prmtop):
    """Charges are in electron units (not AMBER internal units ×18.2223)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    charges = np.asarray(frame["atoms"]["charge"], dtype=float)
    # Raw AMBER value for Li+ would be ~18.22, converted should be ~1.0
    assert all(abs(q) < 5.0 for q in charges), (
        "Charges look like raw AMBER units, not electron units"
    )


# ---------------------------------------------------------------------------
# LJ parameter correctness tests
# ---------------------------------------------------------------------------


def test_nonbond_sigma_epsilon_formula():
    """sigma and epsilon computed from A/B: rMin=(2A/B)^(1/6), eps=0.25B²/A, sigma=2^(-1/6)*rMin."""
    import math

    # F-F self-pair: LENNARD_JONES_ACOEF[0]=2.06132451e5, BCOEF[0]=2.24268406e2
    acoef = 2.06132451e5
    bcoef = 2.24268406e2
    rMin = (2 * acoef / bcoef) ** (1 / 6)
    eps = 0.25 * bcoef**2 / acoef
    sigma = 2 ** (-1 / 6) * rMin
    # F has Rmin/2 ≈ 1.75 Å, so sigma ≈ 3.12 Å; epsilon ≈ 0.061 kcal/mol
    assert abs(sigma - 3.117) < 0.01
    assert abs(eps - 0.061) < 0.002


def test_nonbond_zero_acoef_handled():
    """ZeroDivisionError when acoef==0 is caught; returns rMin=1.0, epsilon=0.0."""
    reader = AmberPrmtopReader("dummy")
    reader.meta = {"NATOM": 1, "NTYPES": 1}
    reader.raw_data = {
        "HBOND_ACOEF": [],
        "HBOND_BCOEF": [],
        "ATOM_TYPE_INDEX": [1],
        "NONBONDED_PARM_INDEX": [1],
        "LENNARD_JONES_ACOEF": [0.0],
        "LENNARD_JONES_BCOEF": [0.0],
    }
    result = reader.parse_nonbond_params({"type": ["DU"]})
    assert len(result) == 1
    _, sigma, epsilon = result[0]
    assert epsilon == 0.0


def test_nonbond_li_sigma_epsilon(litfsi_prmtop):
    """Li+ LJ parameters are physically reasonable (sigma ≈ 1.82 Å, ε ≈ 0.028 kcal/mol)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    atoms = frame["atoms"]
    params = reader.parse_nonbond_params(atoms)
    # Li is the last atom (index 15, 0-based)
    li_params = [p for p in params if p[0] == 16]  # 1-based index 16
    assert li_params, "Li+ params not found"
    _, sigma, epsilon = li_params[0]
    assert 1.5 < sigma < 2.2, f"Li+ sigma={sigma} outside expected range [1.5, 2.2] Å"
    assert 0.01 < epsilon < 0.1, (
        f"Li+ epsilon={epsilon} outside expected range [0.01, 0.1] kcal/mol"
    )


def test_nonbond_f_sigma_epsilon(litfsi_prmtop):
    """F LJ parameters are physically reasonable (sigma ≈ 3.12 Å, ε ≈ 0.061 kcal/mol)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    atoms = frame["atoms"]
    params = reader.parse_nonbond_params(atoms)
    # Atom index 1 (1-based) is F with type_index=1
    f_params = [p for p in params if p[0] == 1]
    assert f_params, "F params not found"
    _, sigma, epsilon = f_params[0]
    assert abs(sigma - 3.117) < 0.05
    assert abs(epsilon - 0.061) < 0.005


def test_nonbond_10_12_raises():
    """Non-zero HBOND coefficients raise an Exception."""
    reader = AmberPrmtopReader("dummy")
    reader.meta = {"NATOM": 1, "NTYPES": 1}
    reader.raw_data = {
        "HBOND_ACOEF": ["1000.0"],
        "HBOND_BCOEF": ["0.5"],
        "ATOM_TYPE_INDEX": [1],
        "NONBONDED_PARM_INDEX": [1],
        "LENNARD_JONES_ACOEF": [1.0],
        "LENNARD_JONES_BCOEF": [1.0],
    }
    with pytest.raises(Exception, match="10-12 interactions"):
        reader.parse_nonbond_params({"type": ["HA"]})


# ---------------------------------------------------------------------------
# Residue assignment correctness tests
# ---------------------------------------------------------------------------


def test_residue_atom_assignment_fsi(litfsi_prmtop):
    """Atoms 0-14 (FSI) are in residue 0; atom 15 (Li) is in residue 1."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    residues = np.asarray(frame["atoms"]["residue"])
    assert all(residues[:15] == 0), f"FSI atoms not all in residue 0: {residues[:15]}"
    assert residues[15] == 1, f"Li+ not in residue 1: {residues[15]}"


def test_residue_count(litfsi_prmtop):
    """LiTFSI has exactly 2 residues (FSI and Li)."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    n_residues = len(np.unique(np.asarray(frame["atoms"]["residue"])))
    assert n_residues == 2


def test_bond_residue_intra_fsi(litfsi_prmtop):
    """Bonds within FSI have residue=0; no bond spans FSI↔Li in LiTFSI."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    bond_residues = np.asarray(frame["bonds"]["residue"])
    # LiTFSI has no Li-FSI bonds, so all bonds are intra-FSI (residue 0)
    assert all(bond_residues == 0), (
        f"Unexpected bond residue values: {np.unique(bond_residues)}"
    )


def test_angle_residue_intra_fsi(litfsi_prmtop):
    """All 25 angles are within FSI (residue 0); none involve Li."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    angle_residues = np.asarray(frame["angles"]["residue"])
    assert all(angle_residues == 0), (
        f"Unexpected angle residue values: {np.unique(angle_residues)}"
    )


# ---------------------------------------------------------------------------
# Title preservation test
# ---------------------------------------------------------------------------


def test_title_preserved_in_metadata(litfsi_prmtop):
    """TITLE section value survives POINTERS section overwriting self.meta."""
    reader = AmberPrmtopReader(litfsi_prmtop)
    frame = Frame()
    frame, _ = reader.read(frame)
    assert "title" in frame.metadata
    assert frame.metadata["title"] == "TFSI"


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_missing_pointers_raises_valueerror(tmp_path):
    """prmtop without a POINTERS section raises ValueError with clear message."""
    prmtop = tmp_path / "no_pointers.prmtop"
    prmtop.write_text(
        "%VERSION  VERSION_STAMP = V0001.000\n%FLAG TITLE\n%FORMAT(20a4)\ntest\n"
    )
    reader = AmberPrmtopReader(prmtop)
    with pytest.raises(ValueError, match="POINTERS section missing"):
        reader.read(Frame())


def test_nonexistent_file_raises():
    """Reading a nonexistent file raises FileNotFoundError."""
    reader = AmberPrmtopReader("/nonexistent/path/file.prmtop")
    with pytest.raises(FileNotFoundError):
        reader.read(Frame())
