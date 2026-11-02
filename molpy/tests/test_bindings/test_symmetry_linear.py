"""Python-binding coverage for ``assign_symmetry_classes``, ``assign_small_rings``
and ``perceive_linear_dihedrals``.

The algorithms are exercised in Rust; these tests assert the PyO3 surface: the
prop written on the atoms and the atom views returned.
"""

import molpy as mp


def _propyne():
    """Propyne CH3-C#CH with explicit hydrogens: C0 C1 C2, H3 on C2, H4-H6 on C0."""
    mol = mp.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=z, h_count=0) for z in (6, 6, 6, 1, 1, 1, 1)]
    mol.add_bond(atoms[0], atoms[1], bond_number=1)
    mol.add_bond(atoms[1], atoms[2], bond_number=3)
    mol.add_bond(atoms[2], atoms[3], bond_number=1)
    for h in atoms[4:]:
        mol.add_bond(atoms[0], h, bond_number=1)
    return mol, atoms


def test_assign_symmetry_classes_writes_the_first_member_of_each_class():
    mol, atoms = _propyne()
    mp.perceive.assign_symmetry_classes(mol)
    assert [atom["symmetry_class"] for atom in atoms] == [0, 1, 2, 3, 4, 4, 4]


def test_perceive_linear_dihedrals_returns_the_molecules_own_atoms():
    mol, atoms = _propyne()
    found = mp.perceive.perceive_linear_dihedrals(mol)
    index = {atom: i for i, atom in enumerate(atoms)}
    quads = sorted(tuple(index[atom] for atom in quad) for quad in found)
    # Every torsion of propyne turns about C1 or C2, both sp: H-C0-C1-C2 (x3) and C0-C1-C2-H3.
    assert quads == [(0, 1, 2, 3), (2, 1, 0, 4), (2, 1, 0, 5), (2, 1, 0, 6)]


def test_assign_small_rings_writes_the_size_on_atoms_and_bonds_within_the_bound():
    """Methylcyclopropane: the ring atoms and bonds read 3, the methyl 0."""
    mol = mp.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=6, h_count=0) for _ in range(4)]
    ring = [mol.add_bond(atoms[i], atoms[j], bond_number=1) for i, j in ((0, 1), (1, 2), (2, 0))]
    methyl = mol.add_bond(atoms[0], atoms[3], bond_number=1)
    mp.perceive.assign_small_rings(mol, 8)
    assert [atom["small_ring_size"] for atom in atoms] == [3, 3, 3, 0]
    assert [bond["small_ring_size"] for bond in ring] == [3, 3, 3]
    assert methyl["small_ring_size"] == 0
