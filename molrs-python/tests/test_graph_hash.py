"""Python-binding coverage for the Weisfeiler-Lehman structural graph hash.

Exercises the PyO3 surface of the native ``graph_hash`` primitive
(``molrs/src/core/graph_hash.rs``) exposed on ``Atomistic`` and
``CoarseGrain``:

- ``structural_hash()`` — isomorphism-invariant 64-bit dedup key,
- ``canonical_order()`` — deterministic node ordering (a ``Refs`` of views),
- ``is_isomorphic(other)`` — whole-graph labeled isomorphism.

The refinement algorithm itself is unit-tested in Rust; these tests assert the
binding surface and the acceptance criteria (ac-001..ac-004).
"""

import molrs

Z = {"H": 1, "C": 6, "O": 8, "S": 16}

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _ethanol(order=(0, 1, 2, 3, 4, 5, 6, 7, 8)):
    """Ethanol skeleton C0-C1-O2 with explicit H, inserted in ``order``.

    Bonds (by original index): C0-C1, C1-O2, O2-H8, C0-H3/H4/H5, C1-H6/H7.
    ``order`` permutes the *node insertion* order to build an isomorphic copy
    whose handles differ but whose graph is identical.
    """
    elems = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    mol = molrs.core.Atomistic()
    atom = {}
    for orig in order:
        atom[orig] = mol.add_atom(atomic_number=Z[elems[orig]])
    for i, j in [(0, 1), (1, 2), (2, 8), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7)]:
        mol.add_bond(atom[i], atom[j])
    return mol, atom


def _linear_chain(n=9, element="C"):
    """A linear ``n``-atom chain (same atom count as ethanol, different topo)."""
    mol = molrs.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=Z[element]) for _ in range(n)]
    for k in range(n - 1):
        mol.add_bond(atoms[k], atoms[k + 1])
    return mol, atoms


# ---------------------------------------------------------------------------
# ac-001: structural_hash is isomorphism-invariant (node-order independent)
# ---------------------------------------------------------------------------


def test_structural_hash_is_int():
    mol, _ = _ethanol()
    h = mol.structural_hash()
    assert isinstance(h, int)
    assert h >= 0


def test_permuted_copy_hashes_equal():
    a, _ = _ethanol()
    b, _ = _ethanol(order=(8, 2, 1, 0, 7, 6, 5, 4, 3))
    assert a.structural_hash() == b.structural_hash()
    assert a.is_isomorphic(b) is True
    assert b.is_isomorphic(a) is True


def test_canonical_order_returns_atom_refs():
    mol, _ = _ethanol()
    order = mol.canonical_order()
    assert isinstance(order, molrs.core.Refs)
    assert len(order) == 9
    # A bijection over the live atoms (no repeats, all real).
    assert len({atom.handle for atom in order}) == 9
    assert set(order) == set(mol.atoms)


# ---------------------------------------------------------------------------
# ac-002: identical junctions equal; a label/topology change differs
# ---------------------------------------------------------------------------


def test_identical_junctions_hash_equal():
    a, _ = _ethanol()
    b, _ = _ethanol()
    assert a.structural_hash() == b.structural_hash()
    assert a.is_isomorphic(b)


def test_element_change_changes_hash():
    a, _ = _ethanol()
    # Replace O(2) with S -> a different local environment.
    b = molrs.core.Atomistic()
    elems = ["C", "C", "S", "H", "H", "H", "H", "H", "H"]
    atoms = [b.add_atom(atomic_number=Z[e]) for e in elems]
    for i, j in [(0, 1), (1, 2), (2, 8), (0, 3), (0, 4), (0, 5), (1, 6), (1, 7)]:
        b.add_bond(atoms[i], atoms[j])
    assert a.structural_hash() != b.structural_hash()
    assert a.is_isomorphic(b) is False


def test_formal_charge_changes_identity_but_partial_charge_does_not():
    a, _ha = _ethanol()
    b, hb = _ethanol()
    assert a.structural_hash() == b.structural_hash()
    hb[2]["charge"] = -1.0
    assert a.structural_hash() == b.structural_hash()
    assert a.is_isomorphic(b)
    hb[2]["formal_charge"] = -1
    assert a.structural_hash() != b.structural_hash()
    assert not a.is_isomorphic(b)


def test_aromatic_flag_change_changes_hash():
    a, _ha = _ethanol()
    b, hb = _ethanol()
    assert a.structural_hash() == b.structural_hash()
    hb[0]["is_aromatic"] = True
    assert a.structural_hash() != b.structural_hash()


def test_bond_class_change_changes_hash():
    a, _ha = _ethanol()
    b, hb = _ethanol()
    assert a.structural_hash() == b.structural_hash()
    # Make the C0-C1 bond a double bond via its relation handle.
    bond = b.bond_between(hb[0], hb[1])
    b.set_bond_class(bond, molrs.core.BondOrder.Double, molrs.core.BondNumber.Double)
    assert a.structural_hash() != b.structural_hash()


def test_kekule_phase_change_changes_hash():
    """Both facts enter the hash, not just the class.

    Two molecules differing only in which ring bonds carry the localized double
    are different Lewis structures, and the hash must say so.
    """
    a, ha = _ethanol()
    b, hb = _ethanol()
    aromatic = molrs.core.BondOrder.Aromatic
    (_other, bond) = b.neighbor_bonds(hb[0])[0]
    b.set_bond_class(bond, aromatic, molrs.core.BondNumber.Single)
    (_other, bond_a) = a.neighbor_bonds(ha[0])[0]
    a.set_bond_class(bond_a, aromatic, molrs.core.BondNumber.Double)
    assert a.structural_hash() != b.structural_hash()


def test_non_isomorphic_graphs_rejected():
    a, _ = _ethanol()
    chain, _ = _linear_chain(9)
    # Same atom count, different connectivity/labels.
    assert a.is_isomorphic(chain) is False


# ---------------------------------------------------------------------------
# ac-003: canonical_order induces a consistent node bijection
# ---------------------------------------------------------------------------


def test_canonical_order_consistent_bijection():
    a, _ = _ethanol()
    b, _ = _ethanol(order=(3, 6, 0, 8, 2, 5, 1, 7, 4))
    oa = a.canonical_order()
    ob = b.canonical_order()
    assert len(oa) == len(ob) == 9
    # Pairing position-by-position yields matching species + degree.
    for ia, ib in zip(oa, ob):
        assert ia["atomic_number"] == ib["atomic_number"]
        assert len(a.neighbor_bonds(ia)) == len(b.neighbor_bonds(ib))


# ---------------------------------------------------------------------------
# ac-004: CoarseGrain bead graph hashes by bead type + topology
# ---------------------------------------------------------------------------


def _cg_angular(types=("W", "P", "W"), bond_order=((0, 1), (1, 2))):
    cg = molrs.core.CoarseGrain()
    beads = [cg.add_bead(bead_type=t) for t in types]
    for i, j in bond_order:
        cg.add_bond(beads[i], beads[j])
    return cg, beads


def test_cg_structural_hash_is_int():
    cg, _ = _cg_angular()
    h = cg.structural_hash()
    assert isinstance(h, int)


def test_cg_identical_bead_graphs_hash_equal():
    cg1, _ = _cg_angular()
    # Same molecule, bonds added in a different order.
    cg2, _ = _cg_angular(bond_order=((1, 2), (0, 1)))
    assert cg1.structural_hash() == cg2.structural_hash()
    assert cg1.is_isomorphic(cg2)


def test_cg_bead_type_change_changes_hash():
    cg1, _ = _cg_angular(types=("W", "P", "W"))
    cg2, _ = _cg_angular(types=("W", "Q", "W"))  # different centre type
    assert cg1.structural_hash() != cg2.structural_hash()
    assert cg1.is_isomorphic(cg2) is False


def test_cg_canonical_order_returns_handles():
    cg, beads = _cg_angular()
    order = cg.canonical_order()
    assert isinstance(order, molrs.core.Refs)
    assert set(order) == set(beads)
