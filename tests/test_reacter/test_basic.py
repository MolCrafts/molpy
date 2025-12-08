"""
Unit tests for the Reacter module.

Tests verify:
- Correct atom removal and bond formation
- Stable atom indexing after merge
- Composability of different selector functions
"""

import pytest

from molpy.core.atomistic import Atom, Atomistic, Bond
from molpy.reacter import (
    ReactionResult,
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
    # Create structure with port marked on atom
    struct = Atomistic()
    c = Atom(symbol="C")
    struct.add_entity(c)
    c["port"] = "1"

    # Select anchor
    anchor = select_port_atom(struct, "1")
    assert anchor is c


def test_remove_one_H():
    """Test removing single hydrogen."""
    # Create C with 3 H
    struct = Atomistic()
    c = Atom(symbol="C")
    h1 = Atom(symbol="H")
    h2 = Atom(symbol="H")
    h3 = Atom(symbol="H")

    struct.add_entity(c, h1, h2, h3)
    struct.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

    # Remove one H
    leaving = select_one_hydrogen(struct, c)
    assert len(leaving) == 1
    assert leaving[0].get("symbol") == "H"


def test_remove_all_H():
    """Test removing all hydrogens."""
    # Create C with 3 H
    struct = Atomistic()
    c = Atom(symbol="C")
    h1 = Atom(symbol="H")
    h2 = Atom(symbol="H")
    h3 = Atom(symbol="H")

    struct.add_entity(c, h1, h2, h3)
    struct.add_link(Bond(c, h1), Bond(c, h2), Bond(c, h3))

    # Remove all H
    leaving = select_all_hydrogens(struct, c)
    assert len(leaving) == 3
    assert all(h.get("symbol") == "H" for h in leaving)


def test_no_leaving_group():
    """Test no leaving group selector."""
    struct = Atomistic()
    c = Atom(symbol="C")
    struct.add_entity(c)

    leaving = select_none(struct, c)
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
    # Create left structure: C-H
    struct_L = Atomistic()
    c_L = Atom(symbol="C")
    h_L = Atom(symbol="H")
    struct_L.add_entity(c_L, h_L)
    struct_L.add_link(Bond(c_L, h_L))
    c_L["port"] = "1"

    # Create right structure: C-H
    struct_R = Atomistic()
    c_R = Atom(symbol="C")
    h_R = Atom(symbol="H")
    struct_R.add_entity(c_R, h_R)
    struct_R.add_link(Bond(c_R, h_R))
    c_R["port"] = "2"

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
    result = cc_coupling.run(struct_L, struct_R, port_L="1", port_R="2")

    # Check product
    assert isinstance(result, ReactionResult)
    assert isinstance(result.product, Atomistic)

    # Should have 2 C + 0 H (both removed)
    atoms = list(result.product.atoms)
    assert len(atoms) == 2

    # Should have 1 C-C bond
    bonds = list(result.product.bonds)
    assert len(bonds) == 1

    # Check result attributes
    assert len(result.removed_atoms) == 2
    assert result.reaction_name == "C-C_coupling"


def test_asymmetric_reaction():
    """Test reaction with different leaving groups."""
    # Create left: C-H-H
    struct_L = Atomistic()
    c_L = Atom(symbol="C")
    h_L1 = Atom(symbol="H")
    h_L2 = Atom(symbol="H")
    struct_L.add_entity(c_L, h_L1, h_L2)
    struct_L.add_link(Bond(c_L, h_L1), Bond(c_L, h_L2))
    c_L["port"] = "1"

    # Create right: C-H
    struct_R = Atomistic()
    c_R = Atom(symbol="C")
    h_R = Atom(symbol="H")
    struct_R.add_entity(c_R, h_R)
    struct_R.add_link(Bond(c_R, h_R))
    c_R["port"] = "2"

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
    result = reacter.run(struct_L, struct_R, port_L="1", port_R="2")

    # Should remove 3 H total
    assert len(result.removed_atoms) == 3

    # Check bond order
    bonds = list(result.product.bonds)
    assert len(bonds) == 1
    assert bonds[0].get("order") == 2


def test_addition_reaction():
    """Test addition reaction (no leaving groups)."""
    # Create two structures
    struct_L = Atomistic()
    c_L = Atom(symbol="C")
    struct_L.add_entity(c_L)
    c_L["port"] = "1"

    struct_R = Atomistic()
    c_R = Atom(symbol="C")
    struct_R.add_entity(c_R)
    c_R["port"] = "2"

    # Addition reaction (no leaving groups)
    addition = Reacter(
        name="addition",
        port_selector_left=select_port_atom,
        port_selector_right=select_port_atom,
        leaving_selector_left=select_none,
        leaving_selector_right=select_none,
        bond_former=form_single_bond,
    )

    result = addition.run(struct_L, struct_R, port_L="1", port_R="2")

    # No atoms removed
    assert len(result.removed_atoms) == 0

    # 2 atoms total
    assert len(list(result.product.atoms)) == 2

    # 1 bond
    assert len(list(result.product.bonds)) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
