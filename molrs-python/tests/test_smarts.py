"""Python-binding coverage for the native SMARTS matcher + graph-edit conveniences.

Exposes the validated Rust ``SmartsPattern`` (``crates/molrs/src/core/chem/smarts``) to
Python with map-keyed results (the molecule's own atom views), plus the graph
edits a reaction workflow leans on (``del_atom`` / ``del_bond`` /
``set_bond_class`` / ``copy.copy``) on ``Atomistic``. The matching algorithm
itself is exercised in Rust; these tests only assert the PyO3 surface and
Daylight atom-map semantics.
"""

import copy

import molrs
import pytest
from molrs.core import BondNumber, BondOrder

Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def _atom(mol, symbol, x, y, z):
    """Add an atom of element ``symbol`` at ``(x, y, z)``; return its view."""
    return mol.add_atom(atomic_number=Z[symbol], x=x, y=y, z=z)


def _methanol() -> "molrs.core.Atomistic":
    """Methanol (CH3-OH) with explicit hydrogens.

    Connectivity: C-O, O-H(hydroxyl), plus three C-H. The C-O-H path is the
    single ``[C][O][H]`` embedding.
    """
    mol = molrs.core.Atomistic()
    c = _atom(mol, "C", 0.0, 0.0, 0.0)
    o = _atom(mol, "O", 1.4, 0.0, 0.0)
    ho = _atom(mol, "H", 2.0, 0.0, 0.0)
    mol.add_bond(c, o)
    mol.add_bond(o, ho)
    for pos in [(-0.4, 1.0, 0.0), (-0.4, -0.5, 0.9), (-0.4, -0.5, -0.9)]:
        h = _atom(mol, "H", *pos)
        mol.add_bond(c, h)
    return mol, c, o, ho


def _ethane() -> "molrs.core.Atomistic":
    """Ethane (C2H6) with explicit hydrogens (no O — for non-match tests)."""
    mol = molrs.core.Atomistic()
    c1 = _atom(mol, "C", 0.0, 0.0, 0.0)
    c2 = _atom(mol, "C", 1.54, 0.0, 0.0)
    for c, (x, y, z) in [
        (c1, (-0.36, 1.03, 0.0)),
        (c1, (-0.36, -0.51, 0.89)),
        (c1, (-0.36, -0.51, -0.89)),
        (c2, (1.90, 1.03, 0.0)),
        (c2, (1.90, -0.51, 0.89)),
        (c2, (1.90, -0.51, -0.89)),
    ]:
        h = _atom(mol, "H", x, y, z)
        mol.add_bond(c, h)
    mol.add_bond(c1, c2)
    return mol, c1, c2


def _methylamine() -> "molrs.core.Atomistic":
    """Methylamine (CH3-NH2) with explicit hydrogens; N carries 2 explicit H."""
    mol = molrs.core.Atomistic()
    c = _atom(mol, "C", 0.0, 0.0, 0.0)
    n = _atom(mol, "N", 1.47, 0.0, 0.0)
    mol.add_bond(c, n)
    for pos in [(1.9, 0.9, 0.0), (1.9, -0.9, 0.0)]:  # 2 H on N
        h = _atom(mol, "H", *pos)
        mol.add_bond(n, h)
    for pos in [(-0.4, 1.0, 0.0), (-0.4, -0.5, 0.9), (-0.4, -0.5, -0.9)]:  # 3 H on C
        h = _atom(mol, "H", *pos)
        mol.add_bond(c, h)
    return mol, n


# ---------------------------------------------------------------------------
# ac-001: SmartsPattern exposed; atom-map capture correct
# ---------------------------------------------------------------------------


def test_smarts_pattern_is_exposed():
    """molrs.perceive.SmartsPattern exists and parses a query."""
    assert "SmartsPattern" in molrs.perceive.__all__
    pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
    assert pat is not None


def test_mapping_captures_map_numbers():
    """A match's mapping is keyed by the query's map numbers."""
    mol, c, o, ho = _methanol()
    pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
    matches = pat.find_matches(mol)
    assert isinstance(matches, list)
    assert len(matches) == 1
    m = matches[0].mapping
    assert set(m.keys()) == {1, 2, 3}
    # Map numbers name the expected atoms / species.
    assert m[1] is c
    assert m[2] is o
    assert m[3] is ho
    assert m[1]["atomic_number"] == 6
    assert m[2]["atomic_number"] == 8
    assert m[3]["atomic_number"] == 1


def test_default_matches_are_objects():
    """find_matches() returns SmartsMatch objects by default."""
    mol, c, o, ho = _methanol()
    pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
    matches = pat.find_matches(mol)
    assert len(matches) == 1
    match = matches[0]
    assert isinstance(match, molrs.perceive.SmartsMatch)
    assert match.atoms == [c, o, ho]
    assert match.mapping == {1: c, 2: o, 3: ho}


def test_no_match_is_empty():
    """A molecule lacking the group yields an empty match list."""
    mol, _c1, _c2 = _ethane()
    pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
    assert pat.find_matches(mol) == []
    assert pat.has_match(mol) is False


def test_primary_amine_matches_NH2():
    """[N;H2:1] matches a primary amine nitrogen (2 explicit H)."""
    mol, n = _methylamine()
    pat = molrs.perceive.SmartsPattern("[N;H2:1]")
    matches = pat.find_matches(mol)
    assert len(matches) == 1
    assert matches[0].mapping == {1: n}
    assert pat.find_matches(mol, root=n)[0].atoms == [n]
    # Ethane (no N) does not match.
    ethane, _c1, _c2 = _ethane()
    assert pat.find_matches(ethane) == []


# ---------------------------------------------------------------------------
# ac-002: Daylight atom-map semantics — map adds no match constraint
# ---------------------------------------------------------------------------


def test_map_number_adds_no_constraint():
    """find_matches('[C:1]') and find_matches('[C]') return the same atom set."""
    mol, c1, c2 = _ethane()
    mapped = molrs.perceive.SmartsPattern("[C:1]").find_matches(mol)
    plain = molrs.perceive.SmartsPattern("[C]").find_matches(mol)
    # Each match is a single-atom SmartsMatch; compare the flat atom sets.
    assert {m.atoms[0] for m in mapped} == {m.atoms[0] for m in plain} == {c1, c2}


def test_map_label_and_num_query_atoms():
    """map_label / n_query_atoms expose the parsed query metadata."""
    pat = molrs.perceive.SmartsPattern("[C:1][O:2][H:3]")
    assert pat.n_query_atoms == 3
    assert pat.map_label(0) == 1
    assert pat.map_label(1) == 2
    assert pat.map_label(2) == 3

    plain = molrs.perceive.SmartsPattern("[C]")
    assert plain.n_query_atoms == 1
    assert plain.map_label(0) is None


def test_labels_and_root_take_atoms_of_the_molecule():
    """``labels`` keys and ``root`` are the molecule's atoms; an atom of
    another molecule is refused."""
    mol, c, o, ho = _methanol()
    pat = molrs.perceive.SmartsPattern("[C:1][O:2]")
    assert [m.atoms for m in pat.find_matches(mol, root=c)] == [[c, o]]
    assert pat.find_matches(mol, root=o) == []
    assert pat.has_match(mol, labels={c: "x"})
    other, other_c, _o, _ho = _methanol()
    with pytest.raises(ValueError, match="another graph"):
        pat.find_matches(mol, root=other_c)
    with pytest.raises(TypeError, match="Atom view"):
        pat.find_matches(mol, root=0)


def test_from_environment_takes_the_center_atom():
    mol, c, _o, _ho = _methanol()
    pat = molrs.perceive.SmartsPattern.from_environment(mol, c)
    assert pat.find_matches(mol, root=c)


# ---------------------------------------------------------------------------
# ac-003: Atomistic graph edits
# ---------------------------------------------------------------------------


def test_del_atom_cascades():
    """del_atom drops the atom and its incident bonds."""
    mol, _c, o, _ho = _methanol()
    n_bonds_before = mol.n_bonds
    assert mol.n_atoms == 6
    mol.del_atom(o)  # O is bonded to C and to H → 2 incident bonds cascade
    assert mol.n_atoms == 5
    assert mol.n_bonds == n_bonds_before - 2
    assert all(atom["atomic_number"] != 8 for atom in mol.atoms)


def test_del_bond():
    """del_bond removes just the bond, leaving atoms intact."""
    mol, c, _o, _ho = _methanol()
    (_other, bond) = mol.neighbor_bonds(c)[0]
    n_atoms_before = mol.n_atoms
    n_bonds_before = mol.n_bonds
    mol.del_bond(bond)
    assert mol.n_bonds == n_bonds_before - 1
    assert mol.n_atoms == n_atoms_before  # atoms untouched


def test_set_bond_class():
    """set_bond_class writes both facts together: the class and the number."""
    mol, c, _o, _ho = _methanol()
    (_other, bond) = mol.neighbor_bonds(c)[0]
    mol.set_bond_class(bond, BondOrder.Double, BondNumber.Double)
    # Both are written — a class with no number leaves the bond
    # un-standardized.
    assert bond["bond_type"] == BondOrder.Double.code
    assert bond["bond_number"] == BondNumber.Double.code
    # Aromatic is a class whose number is the Kekulé phase, stated apart.
    mol.set_bond_class(bond, BondOrder.Aromatic, BondNumber.Single)
    assert bond["bond_type"] == BondOrder.Aromatic.code
    assert bond["bond_number"] == BondNumber.Single.code


def test_copy_is_independent():
    """copy.copy yields an independent graph — mutating the copy leaves the original."""
    mol, _c, _o, _ho = _methanol()
    n_atoms = mol.n_atoms
    n_bonds = mol.n_bonds
    dup = copy.copy(mol)
    assert dup.n_atoms == n_atoms
    dup.del_atom(dup.atoms[0])
    # Original untouched.
    assert mol.n_atoms == n_atoms
    assert mol.n_bonds == n_bonds
    assert dup.n_atoms == n_atoms - 1
