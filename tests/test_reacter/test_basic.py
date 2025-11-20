"""
Unit tests for the Reacter module.

Tests verify:
- Correct atom removal and bond formation
- Stable atom indexing after merge
- Composability of different selector functions
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter import (
    ReactionProduct,
    Reacter,
    find_neighbors,
    form_double_bond,
    form_single_bond,
    select_none,
    select_port_atom,
    select_all_hydrogens,
    select_one_hydrogen,
)


def test_merge_simple():
    """Test merging two simple assemblies using instance method."""
    # Create first assembly: C-H
    asm1 = Atomistic()
    c1 = Atom(symbol="C")
    h1 = Atom(symbol="H")
    b1 = Bond(c1, h1, order=1)
    asm1.add_entity(c1, h1)
    asm1.add_link(b1, include_endpoints=False)  # Don't add endpoints twice

    # Create second assembly: C-H
    asm2 = Atomistic()
    c2 = Atom(symbol="C")
    h2 = Atom(symbol="H")
    b2 = Bond(c2, h2, order=1)
    asm2.add_entity(c2, h2)
    asm2.add_link(b2, include_endpoints=False)  # Don't add endpoints twice

    # Store original for verification
    original_c1 = c1
    original_c2 = c2

    # Merge asm2 into asm1
    result = asm1.merge(asm2)

    # Check that merge returns self
    assert result is asm1

    # Check total atoms
    assert len(list(asm1.atoms)) == 4  # 2 C + 2 H
    assert len(list(asm1.bonds)) == 2  # 2 bonds

    # Check that original atoms are still in asm1
    assert original_c1 in asm1.atoms
    # c2 SHOULD be in asm1 now (no deep copy, direct transfer!)
    assert original_c2 in asm1.atoms


def test_port_anchor_selector():
    """Test selecting anchor from port."""
    # Create monomer with port
    mono = Monomer()
    c = Atom(symbol="C")
    mono.add_entity(c)

    mono.set_port("1", c)

    # Select anchor
    anchor = select_port_atom(mono, "1")
    assert anchor is c


def test_remove_one_H():
    """Test removing single hydrogen."""
    # Create C with 3 H
    mono = Monomer()
    c = Atom(symbol="C")
    h1 = Atom(symbol="H")
    h2 = Atom(symbol="H")
    h3 = Atom(symbol="H")

    mono.add_entity(c, h1, h2, h3)
    mono.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

    # Remove one H
    leaving = select_one_hydrogen(mono, c)
    assert len(leaving) == 1
    assert leaving[0].get("symbol") == "H"


def test_remove_all_H():
    """Test removing all hydrogens."""
    # Create C with 3 H
    mono = Monomer()
    c = Atom(symbol="C")
    h1 = Atom(symbol="H")
    h2 = Atom(symbol="H")
    h3 = Atom(symbol="H")

    mono.add_entity(c, h1, h2, h3)
    mono.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

    # Remove all H
    leaving = select_all_hydrogens(mono, c)
    assert len(leaving) == 3
    assert all(h.get("symbol") == "H" for h in leaving)


def test_no_leaving_group():
    """Test no leaving group selector."""
    mono = Monomer()
    c = Atom(symbol="C")
    mono.add_entity(c)

    leaving = select_none(mono, c)
    assert len(leaving) == 0


def test_find_neighbors():
    """Test finding neighbors of an atom."""
    asm = Atomistic()
    c = Atom(symbol="C")
    h1 = Atom(symbol="H")
    h2 = Atom(symbol="H")
    o = Atom(symbol="O")

    asm.add_entity(c, h1, h2, o)
    asm.add_link(Bond(c, h1), Bond(c, h2), Bond(c, o))

    # Find all neighbors
    neighbors = find_neighbors(asm, c)
    assert len(neighbors) == 3

    # Find only H neighbors
    h_neighbors = find_neighbors(asm, c, element="H")
    assert len(h_neighbors) == 2
    assert all(h.get("symbol") == "H" for h in h_neighbors)


def test_simple_cc_coupling():
    """Test simple C-C coupling reaction."""
    # Create left monomer: C-H
    mono_L = Monomer()
    c_L = Atom(symbol="C")
    h_L = Atom(symbol="H")
    mono_L.add_entity(c_L, h_L)
    mono_L.add_link(Bond(c_L, h_L))
    mono_L.set_port("1", c_L)

    # Create right monomer: C-H
    mono_R = Monomer()
    c_R = Atom(symbol="C")
    h_R = Atom(symbol="H")
    mono_R.add_entity(c_R, h_R)
    mono_R.add_link(Bond(c_R, h_R))
    mono_R.set_port("2", c_R)

    # Create C-C coupling reaction
    cc_coupling = Reacter(
        name="C-C_coupling",
        port_selector_left=select_port_atom,
        port_selector_right=select_port_atom,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
    )

    # Run reaction
    product = cc_coupling.run(mono_L, mono_R, port_L="1", port_R="2")

    # Check product
    assert isinstance(product, ReactionProduct)
    assert isinstance(product.product, Atomistic)

    # Should have 2 C + 0 H (both removed)
    atoms = list(product.product.atoms)
    assert len(atoms) == 2

    # Should have 1 C-C bond
    bonds = list(product.product.bonds)
    assert len(bonds) == 1

    # Check notes
    assert product.notes["n_eliminated"] == 2
    assert product.notes["reaction_name"] == "C-C_coupling"


def test_asymmetric_reaction():
    """Test reaction with different leaving groups."""
    # Create left: C-H-H
    mono_L = Monomer()
    c_L = Atom(symbol="C")
    h_L1 = Atom(symbol="H")
    h_L2 = Atom(symbol="H")
    mono_L.add_entity(c_L, h_L1, h_L2)
    mono_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2))
    mono_L.set_port("1", c_L)

    # Create right: C-H
    mono_R = Monomer()
    c_R = Atom(symbol="C")
    h_R = Atom(symbol="H")
    mono_R.add_entity(c_R, h_R)
    mono_R.add_link(Bond(c_R, h_R))
    mono_R.set_port("2", c_R)

    # Create reaction: remove all H from left, one H from right
    reacter = Reacter(
        name="asymmetric",
        port_selector_left=select_port_atom,
        port_selector_right=select_port_atom,
        leaving_selector_left=select_all_hydrogens,  # Remove both H
        leaving_selector_right=select_one_hydrogen,  # Remove one H
        bond_former=form_double_bond,  # Make C=C
    )

    # Run reaction
    product = reacter.run(mono_L, mono_R, port_L="1", port_R="2")

    # Should remove 3 H total
    assert product.notes["n_eliminated"] == 3

    # Check bond order
    bonds = list(product.product.bonds)
    assert len(bonds) == 1
    assert bonds[0].get("order") == 2


def test_addition_reaction():
    """Test addition reaction (no leaving groups)."""
    # Create two monomers
    mono_L = Monomer()
    c_L = Atom(symbol="C")
    mono_L.add_entity(c_L)
    mono_L.set_port("1", c_L)

    mono_R = Monomer()
    c_R = Atom(symbol="C")
    mono_R.add_entity(c_R)
    mono_R.set_port("2", c_R)

    # Addition reaction (no leaving groups)
    addition = Reacter(
        name="addition",
        port_selector_left=select_port_atom,
        port_selector_right=select_port_atom,
        leaving_selector_left=select_none,
        leaving_selector_right=select_none,
        bond_former=form_single_bond,
    )

    product = addition.run(mono_L, mono_R, port_L="1", port_R="2")

    # No atoms removed
    assert product.notes["n_eliminated"] == 0

    # 2 atoms total
    assert len(list(product.product.atoms)) == 2

    # 1 bond
    assert len(list(product.product.bonds)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
