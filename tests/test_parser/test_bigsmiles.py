"""Tests for BigSMILES parser.

This module contains comprehensive tests for BigSMILES parsing, including:
- Bond extraction and matching
- Port extraction from descriptors
- Monomer conversion
- Edge cases and fix verification
"""

import pytest

from molpy.parser.smiles import parse_bigsmiles, bigsmilesir_to_monomer
from molpy.core.atomistic import Atomistic


class TestBondExtraction:
    """Test bond extraction from BigSMILES strings."""

    def test_pegdae3_all_bonds_extracted(self):
        """Test that all 15 bonds are extracted from PEGDAE3 monomer."""
        bigsmiles = "{[<]C=CCOCCOCCOCCOCC=C[>]}"

        # Parse BigSMILES
        ir = parse_bigsmiles(bigsmiles)

        # Extract graph
        stoch_obj = ir.stochastic_objects[0]
        repeat_unit = stoch_obj.repeat_units[0]
        graph = repeat_unit.graph

        # Build atom mapping using id() for reliable matching
        atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(graph.atoms)}

        # Collect valid bonds
        valid_bonds = []
        for bond in graph.bonds:
            i_idx = atom_ir_to_idx.get(id(bond.atom_i))
            j_idx = atom_ir_to_idx.get(id(bond.atom_j))

            if i_idx is not None and j_idx is not None and i_idx != j_idx:
                valid_bonds.append((i_idx, j_idx, bond.order))

        # Check unique bonds
        unique_bonds = {}
        for i, j, order in valid_bonds:
            key = tuple(sorted([i, j]))
            if key not in unique_bonds:
                unique_bonds[key] = order

        # Expected bonds for C=CCOCCOCCOCCOCC=C (16 atoms, 15 bonds)
        # C=C-C-O-C-C-O-C-C-O-C-C-O-C-C=C
        # 0=1-2-3-4-5-6-7-8-9-10-11-12-13-14=15
        expected_bonds = {
            (0, 1): 2,  # C=C (double)
            (1, 2): 1,  # C-C
            (2, 3): 1,  # C-O
            (3, 4): 1,  # O-C
            (4, 5): 1,  # C-C
            (5, 6): 1,  # C-O
            (6, 7): 1,  # O-C
            (7, 8): 1,  # C-C
            (8, 9): 1,  # C-O
            (9, 10): 1,  # O-C
            (10, 11): 1,  # C-C
            (11, 12): 1,  # C-O
            (12, 13): 1,  # O-C
            (13, 14): 1,  # C-C
            (14, 15): 2,  # C=C (double)
        }

        assert len(unique_bonds) == len(
            expected_bonds
        ), f"Expected {len(expected_bonds)} bonds, got {len(unique_bonds)}"

        # Verify each expected bond
        for (i, j), expected_order in expected_bonds.items():
            key = tuple(sorted([i, j]))
            found_order = unique_bonds.get(key)
            assert (
                found_order is not None
            ), f"Bond {i}-{j} (order={expected_order}) not found"
            assert (
                found_order == expected_order
            ), f"Bond {i}-{j} has wrong order: expected {expected_order}, got {found_order}"

    def test_simple_double_bond(self):
        """Test simple double bond extraction."""
        bigsmiles = "{[<]C=C[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert len(monomer.atoms) == 2
        assert len(monomer.bonds) == 1

        # Check bond order
        bond = list(monomer.bonds)[0]
        assert (
            bond.get("kind") == 2
        ), f"Expected double bond (order=2), got {bond.get('kind')}"

    def test_simple_single_bond(self):
        """Test simple single bond extraction."""
        bigsmiles = "{[<]CC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert len(monomer.atoms) == 2
        assert len(monomer.bonds) == 1

        # Check bond order
        bond = list(monomer.bonds)[0]
        assert (
            bond.get("kind") == 1
        ), f"Expected single bond (order=1), got {bond.get('kind')}"

    def test_no_false_self_bonds(self):
        """Test that atoms with same element but different identity are not treated as self-bonds."""
        bigsmiles = "{[<]CCO[>]}"

        ir = parse_bigsmiles(bigsmiles)
        stoch_obj = ir.stochastic_objects[0]
        repeat_unit = stoch_obj.repeat_units[0]
        graph = repeat_unit.graph

        # Build atom mapping
        atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(graph.atoms)}

        # Check all bonds are valid (no self-bonds)
        for bond in graph.bonds:
            i_idx = atom_ir_to_idx.get(id(bond.atom_i))
            j_idx = atom_ir_to_idx.get(id(bond.atom_j))

            assert (
                i_idx is not None and j_idx is not None
            ), "Bond references atoms not in graph"
            assert (
                i_idx != j_idx
            ), f"Self-bond detected: atom {i_idx} connected to itself"

    def test_ether_chain_bonds(self):
        """Test bond extraction in ether chain (C-O-C)."""
        bigsmiles = "{[<]COC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert len(monomer.atoms) == 3
        assert len(monomer.bonds) == 2

        # Verify connectivity: C-O-C
        atoms = list(monomer.atoms)
        bonds = list(monomer.bonds)

        # Find O atom
        o_atom = next(atom for atom in atoms if atom.get("symbol") == "O")

        # Count bonds to O (should be 2)
        o_bonds = [b for b in bonds if o_atom in (b.itom, b.jtom)]
        assert len(o_bonds) == 2, f"O atom should have 2 bonds, got {len(o_bonds)}"

    def test_long_chain_bonds(self):
        """Test bond extraction in longer chain."""
        bigsmiles = "{[<]CCCCCC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        # 6 carbons, 5 bonds
        assert len(monomer.atoms) == 6
        assert len(monomer.bonds) == 5

        # All bonds should be single bonds
        for bond in monomer.bonds:
            assert (
                bond.get("kind") == 1
            ), f"Expected single bond, got {bond.get('kind')}"

    def test_mixed_bond_orders(self):
        """Test chain with mixed single and double bonds."""
        bigsmiles = "{[<]C=CC=CC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert len(monomer.atoms) == 5
        assert len(monomer.bonds) == 4

        # Count double bonds
        double_bonds = [b for b in monomer.bonds if b.get("kind") == 2]
        assert (
            len(double_bonds) == 2
        ), f"Expected 2 double bonds, got {len(double_bonds)}"

        # Count single bonds
        single_bonds = [b for b in monomer.bonds if b.get("kind") == 1]
        assert (
            len(single_bonds) == 2
        ), f"Expected 2 single bonds, got {len(single_bonds)}"


class TestMonomerConversion:
    """Test conversion of BigSMILES IR to Atomistic monomers."""

    def test_pegdae3_monomer_conversion(self):
        """Test conversion of PEGDAE3 BigSMILES to Atomistic monomer."""
        bigsmiles = "{[<]C=CCOCCOCCOCCOCC=C[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert isinstance(monomer, Atomistic)
        assert len(monomer.atoms) == 16, f"Expected 16 atoms, got {len(monomer.atoms)}"
        assert len(monomer.bonds) == 15, f"Expected 15 bonds, got {len(monomer.bonds)}"

        # Check ports
        ports = [
            atom.get("port") for atom in monomer.atoms if atom.get("port") is not None
        ]
        assert len(ports) == 2, f"Expected 2 ports, got {len(ports)}"
        assert "<" in ports, "Expected '<' port"
        assert ">" in ports, "Expected '>' port"

    def test_simple_monomer_conversion(self):
        """Test conversion of simple BigSMILES to monomer."""
        bigsmiles = "{[<]CC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert isinstance(monomer, Atomistic)
        assert len(monomer.atoms) == 2
        assert len(monomer.bonds) == 1


class TestPortExtraction:
    """Test port extraction from BigSMILES descriptors."""

    def test_left_right_ports(self):
        """Test that [<] and [>] descriptors create '<' and '>' ports."""
        bigsmiles = "{[<]CC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        ports = {}
        for atom in monomer.atoms:
            port = atom.get("port")
            if port is not None:
                ports[port] = atom.get("symbol")

        assert "<" in ports, "Expected '<' port from [<]"
        assert ">" in ports, "Expected '>' port from [>]"
        assert ports["<"] == "C", "Left port should be on first atom (C)"
        assert ports[">"] == "C", "Right port should be on last atom (C)"

    def test_ports_on_terminal_atoms(self):
        """Test that ports are correctly placed on terminal atoms."""
        bigsmiles = "{[<]CCO[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        atoms = list(monomer.atoms)

        # First atom should have '<' port
        first_atom = atoms[0]
        assert first_atom.get("port") == "<", "First atom should have '<' port"

        # Last atom should have '>' port
        last_atom = atoms[-1]
        assert last_atom.get("port") == ">", "Last atom should have '>' port"

        # Middle atoms should not have ports
        for atom in atoms[1:-1]:
            assert (
                atom.get("port") is None
            ), f"Middle atom {atom.get('symbol')} should not have port"


class TestBondMatchingFix:
    """Test that the bond matching fix works correctly.

    The fix changed bond matching from == (value comparison) to id() (identity comparison)
    to avoid false self-bond detection due to SmilesAtomIR's dataclass equality comparison.
    """

    def test_atoms_with_same_element_not_treated_as_equal(self):
        """
        Test that atoms with same element but different identity
        are correctly distinguished using id() matching.

        This is the core fix: SmilesAtomIR uses @dataclass(eq=True),
        so == compares by value, not identity. We must use id() for matching.
        """
        bigsmiles = "{[<]CCCC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        stoch_obj = ir.stochastic_objects[0]
        repeat_unit = stoch_obj.repeat_units[0]
        graph = repeat_unit.graph

        # All atoms are C, so == would return True for any pair
        # But they are different objects, so id() distinguishes them
        atoms = graph.atoms
        assert len(atoms) == 4

        # Verify all atoms are different objects
        atom_ids = [id(atom) for atom in atoms]
        assert len(set(atom_ids)) == 4, "All atoms should be different objects"

        # Verify all atoms have same element (would be equal by ==)
        for atom in atoms:
            assert atom.element == "C"

        # Verify bonds connect different atoms (by identity)
        atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(atoms)}
        for bond in graph.bonds:
            i_idx = atom_ir_to_idx.get(id(bond.atom_i))
            j_idx = atom_ir_to_idx.get(id(bond.atom_j))
            assert i_idx != j_idx, "Bond should connect different atoms"

        # Convert to monomer and verify
        monomer = bigsmilesir_to_monomer(ir)
        assert len(monomer.bonds) == 3, "Should have 3 bonds for 4 atoms"

    def test_bond_matching_uses_identity_not_value(self):
        """Test that bond matching uses object identity, not value equality."""
        bigsmiles = "{[<]CCOC[>]}"

        ir = parse_bigsmiles(bigsmiles)
        stoch_obj = ir.stochastic_objects[0]
        repeat_unit = stoch_obj.repeat_units[0]
        graph = repeat_unit.graph

        # Build identity-based mapping (as used in fixed converter)
        atom_ir_to_idx = {id(atom_ir): idx for idx, atom_ir in enumerate(graph.atoms)}

        # All bonds should connect different atoms by identity
        for bond in graph.bonds:
            i_idx = atom_ir_to_idx.get(id(bond.atom_i))
            j_idx = atom_ir_to_idx.get(id(bond.atom_j))

            # Both atoms should be found
            assert i_idx is not None, "atom_i should be found by id()"
            assert j_idx is not None, "atom_j should be found by id()"

            # Atoms should be different (by identity)
            assert i_idx != j_idx, "Bond should connect different atoms"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_bigsmiles(self):
        """Test parsing empty BigSMILES string."""
        # Empty string should parse to empty IR
        ir = parse_bigsmiles("")
        # Should not raise error
        assert ir is not None

    def test_single_atom(self):
        """Test BigSMILES with single atom."""
        bigsmiles = "{[<]C[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        assert len(monomer.atoms) == 1
        assert len(monomer.bonds) == 0  # No bonds for single atom

    def test_ring_structure(self):
        """Test BigSMILES with ring structure."""
        # Simple ring: cyclopropane
        bigsmiles = "{[<]C1CC1[>]}"

        ir = parse_bigsmiles(bigsmiles)
        monomer = bigsmilesir_to_monomer(ir)

        # Should have 3 atoms and 3 bonds (ring)
        assert len(monomer.atoms) == 3
        assert len(monomer.bonds) == 3
