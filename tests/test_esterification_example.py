"""
Test esterification reaction example for reacter tutorial.

This tests the reaction: CH3COOH + CH3CH2OH → CH3COOCH2CH3 + H2O

Now using the FIXED remove_OH selector that correctly handles carboxylic acids.
"""

from molpy import Atom, Atomistic, Bond
from molpy.core.entity import Entity
from molpy.core.wrappers.monomer import Monomer
from molpy.reacter import (
    Reacter,
    port_anchor_selector,
    remove_OH,  # Fixed version!
    remove_one_H,
    make_single_bond,
)


def create_acetic_acid():
    """Create acetic acid: CH3-COOH"""
    mono = Monomer(name="acetic_acid")

    # CH3 group
    c1 = Atom(symbol="C", name="C1")
    h1 = Atom(symbol="H", name="H1")
    h2 = Atom(symbol="H", name="H2")
    h3 = Atom(symbol="H", name="H3")

    # COOH group
    c2 = Atom(symbol="C", name="C2")  # Carbonyl carbon
    o1 = Atom(symbol="O", name="O1")  # Carbonyl oxygen (double bond)
    o2 = Atom(symbol="O", name="O2")  # Hydroxyl oxygen
    h4 = Atom(symbol="H", name="H4")  # Hydroxyl hydrogen

    mono.add_entity(c1, h1, h2, h3, c2, o1, o2, h4)

    # Bonds
    mono.add_link(
        Bond(c1, h1, order=1),
        Bond(c1, h2, order=1),
        Bond(c1, h3, order=1),
        Bond(c1, c2, order=1),  # CH3-C
        Bond(c2, o1, order=2),  # C=O (carbonyl)
        Bond(c2, o2, order=1),  # C-OH (hydroxyl)
        Bond(o2, h4, order=1),  # O-H
    )

    # Set port on carbonyl carbon (where ester bond will form)
    mono.set_port("1", c2)

    return mono


def create_ethanol():
    """Create ethanol: CH3-CH2-OH"""
    mono = Monomer(name="ethanol")

    # CH3 group
    c1 = Atom(symbol="C", name="C1")
    h1 = Atom(symbol="H", name="H1")
    h2 = Atom(symbol="H", name="H2")
    h3 = Atom(symbol="H", name="H3")

    # CH2OH group
    c2 = Atom(symbol="C", name="C2")
    h4 = Atom(symbol="H", name="H4")
    h5 = Atom(symbol="H", name="H5")
    o = Atom(symbol="O", name="O")
    h6 = Atom(symbol="H", name="H6")  # This H will be removed

    mono.add_entity(c1, h1, h2, h3, c2, h4, h5, o, h6)

    # Bonds
    mono.add_link(
        Bond(c1, h1, order=1),
        Bond(c1, h2, order=1),
        Bond(c1, h3, order=1),
        Bond(c1, c2, order=1),  # CH3-CH2
        Bond(c2, h4, order=1),
        Bond(c2, h5, order=1),
        Bond(c2, o, order=1),  # CH2-O
        Bond(o, h6, order=1),  # O-H
    )

    # Set port on oxygen (where ester bond will form)
    mono.set_port("1", o)

    return mono


def test_esterification():
    """Test esterification reaction."""
    print("=" * 60)
    print("Esterification Reaction Test")
    print("=" * 60)
    print("\nReaction: CH3COOH + CH3CH2OH → CH3COOCH2CH3 + H2O")
    print("         (acetic acid + ethanol → ethyl acetate + water)\n")

    print("Creating reactants...")
    acetic_acid = create_acetic_acid()
    ethanol = create_ethanol()

    print(f"├─ Acetic acid (CH3COOH): {len(list(acetic_acid.atoms))} atoms")
    print(f"└─ Ethanol (CH3CH2OH): {len(list(ethanol.atoms))} atoms")

    # Create esterification reacter
    esterification = Reacter(
        name="esterification",
        anchor_left=port_anchor_selector,
        anchor_right=port_anchor_selector,
        leaving_left=remove_OH,  # Fixed remove_OH now works correctly!
        leaving_right=remove_one_H,  # Remove -H from alcohol
        bond_maker=make_single_bond,  # Form C-O ester bond
    )

    print("\nRunning esterification reaction...")
    print("├─ Left leaving group: -OH from carboxylic acid")
    print("├─ Right leaving group: -H from alcohol")
    print("└─ New bond: C-O ester linkage\n")

    product = esterification.run(acetic_acid, ethanol, port_L="1", port_R="1")

    print("=" * 60)
    print("Reaction Results")
    print("=" * 60)
    print(f"Reaction name: {product.notes['reaction_name']}")
    print(f"Removed atoms: {product.notes['removed_count']} (forming H2O)")
    print(f"Remaining atoms: {len(list(product.product.atoms))}")
    print(f"Total bonds: {len(list(product.product.bonds))}")

    # Count elements
    atom_counts = {}
    for atom in product.product.atoms:
        symbol = atom.get("symbol")
        atom_counts[symbol] = atom_counts.get(symbol, 0) + 1

    formula = "".join(
        f"{symbol}{atom_counts[symbol] if atom_counts[symbol] > 1 else ''}"
        for symbol in sorted(atom_counts.keys())
        if symbol  # Filter out None
    )

    print(f"\nMolecular formula: {formula}")
    print(f"Expected formula: C4H8O2 (ethyl acetate)")

    # Verify
    expected_atoms = 14  # C4H8O2
    actual_atoms = len(list(product.product.atoms))

    print("\n" + "=" * 60)
    print("Validation")
    print("=" * 60)

    try:
        assert actual_atoms == expected_atoms, (
            f"Expected {expected_atoms} atoms, got {actual_atoms}"
        )
        assert atom_counts.get("C") == 4, "Expected 4 carbons"
        assert atom_counts.get("H") == 8, "Expected 8 hydrogens"
        assert atom_counts.get("O") == 2, "Expected 2 oxygens"
        assert product.notes["removed_count"] == 3, "Expected 3 atoms removed (H2O)"

        print("✓ Atom count correct")
        print("✓ Molecular formula correct (C4H8O2)")
        print("✓ Leaving groups removed correctly (H2O)")
        print("\n🎉 Esterification test PASSED!")

    except AssertionError as e:
        print(f"❌ Test FAILED: {e}")
        raise

    return product


if __name__ == "__main__":
    test_esterification()
