"""Unit tests for RDKitWrapper to verify element symbol handling.

This test suite specifically checks that element symbols are correctly
preserved when adding hydrogens and converting between RDKit and Atomistic.

Focus areas:
1. Symbol preservation after AddHs()
2. Correct mapping between RDKit atoms and Atomistic atoms
3. Hydrogen addition with correct symbols
4. Round-trip conversion (Atomistic -> Mol -> Atomistic)
"""

import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from molpy import Atomistic
from molpy.adapter import RDKitWrapper


class TestRDKitWrapperSymbolHandling:
    """Test element symbol handling in RDKitWrapper."""

    def test_basic_symbol_preservation(self):
        """Test that symbols are preserved in simple conversion."""
        # Create simple molecule: CCO (ethanol)
        mol = Chem.MolFromSmiles("CCO")

        # Convert to Atomistic
        wrapper = RDKitWrapper.from_mol(mol)
        atomistic = wrapper.inner

        # Check symbols
        atoms = list(atomistic.atoms)
        symbols = [atom.get("symbol") for atom in atoms]

        assert len(symbols) == 3, f"Expected 3 atoms, got {len(symbols)}"
        assert symbols == ["C", "C", "O"], f"Expected ['C', 'C', 'O'], got {symbols}"

    def test_symbol_after_add_hydrogens(self):
        """Test that symbols are correct after adding hydrogens with RDKit."""
        # Create molecule without explicit H
        mol = Chem.MolFromSmiles("CCO")
        assert mol.GetNumAtoms() == 3, "Should have 3 heavy atoms"

        # Add hydrogens using RDKit
        mol_h = Chem.AddHs(mol)
        assert mol_h.GetNumAtoms() == 9, "Should have 9 total atoms (3 heavy + 6 H)"

        # Convert to Atomistic
        wrapper = RDKitWrapper.from_mol(mol_h)
        atomistic = wrapper.inner

        # Check all symbols
        atoms = list(atomistic.atoms)
        symbols = [atom.get("symbol") for atom in atoms]

        # Count elements
        c_count = symbols.count("C")
        o_count = symbols.count("O")
        h_count = symbols.count("H")

        assert c_count == 2, f"Expected 2 C atoms, got {c_count}"
        assert o_count == 1, f"Expected 1 O atom, got {o_count}"
        assert h_count == 6, f"Expected 6 H atoms, got {h_count}"

        # Verify atomic numbers match symbols
        for atom in atoms:
            symbol = atom.get("symbol")
            atomic_num = atom.get("atomic_num")

            if symbol == "C":
                assert atomic_num == 6, f"C should have atomic_num=6, got {atomic_num}"
            elif symbol == "O":
                assert atomic_num == 8, f"O should have atomic_num=8, got {atomic_num}"
            elif symbol == "H":
                assert atomic_num == 1, f"H should have atomic_num=1, got {atomic_num}"
            else:
                pytest.fail(f"Unexpected symbol: {symbol}")

    def test_generate_3d_symbol_preservation(self):
        """Test that symbols are preserved after generate_3d()."""
        # Create molecule
        mol = Chem.MolFromSmiles("CCCCO")  # Butanol

        # Create wrapper and generate 3D
        wrapper = RDKitWrapper.from_mol(mol)
        wrapper.generate_3d(optimize=True, add_hydrogens=True)

        atomistic = wrapper.inner
        atoms = list(atomistic.atoms)

        # Expected: 4 C + 1 O + 10 H = 15 atoms
        assert len(atoms) >= 5, "Should have at least 5 heavy atoms"

        # Check heavy atoms
        heavy_atoms = [a for a in atoms if a.get("atomic_num", 0) != 1]
        heavy_symbols = [a.get("symbol") for a in heavy_atoms]

        c_count = heavy_symbols.count("C")
        o_count = heavy_symbols.count("O")

        assert c_count == 4, f"Expected 4 C atoms, got {c_count}"
        assert o_count == 1, f"Expected 1 O atom, got {o_count}"

        # Check that ALL atoms have valid symbols
        for i, atom in enumerate(atoms):
            symbol = atom.get("symbol")
            atomic_num = atom.get("atomic_num")

            assert symbol is not None, f"Atom {i} missing symbol"
            assert atomic_num is not None, f"Atom {i} missing atomic_num"

            # Verify symbol matches atomic number
            expected_symbols = {1: "H", 6: "C", 8: "O"}
            expected_symbol = expected_symbols.get(atomic_num)

            assert (
                symbol == expected_symbol
            ), f"Atom {i}: symbol={symbol} doesn't match atomic_num={atomic_num} (expected {expected_symbol})"

    def test_round_trip_conversion(self):
        """Test Atomistic -> Mol -> Atomistic round-trip."""
        # Create initial Atomistic
        atomistic1 = Atomistic()
        c1 = atomistic1.def_atom(symbol="C", atomic_num=6, xyz=[0.0, 0.0, 0.0])
        c2 = atomistic1.def_atom(symbol="C", atomic_num=6, xyz=[1.5, 0.0, 0.0])
        o1 = atomistic1.def_atom(symbol="O", atomic_num=8, xyz=[3.0, 0.0, 0.0])
        atomistic1.def_bond(c1, c2, order=1.0)
        atomistic1.def_bond(c2, o1, order=1.0)

        # Convert to Mol
        wrapper1 = RDKitWrapper.from_atomistic(atomistic1)
        mol = wrapper1.mol

        # Convert back to Atomistic
        wrapper2 = RDKitWrapper.from_mol(mol)
        atomistic2 = wrapper2.inner

        # Check symbols are preserved
        atoms1 = list(atomistic1.atoms)
        atoms2 = list(atomistic2.atoms)

        symbols1 = [a.get("symbol") for a in atoms1]
        symbols2 = [a.get("symbol") for a in atoms2]

        assert (
            symbols1 == symbols2
        ), f"Symbols changed in round-trip: {symbols1} -> {symbols2}"

    def test_hydrogen_addition_symbols(self):
        """Test that hydrogens added via generate_3d have correct symbols."""
        # Create simple molecule
        mol = Chem.MolFromSmiles("CC")  # Ethane

        wrapper = RDKitWrapper.from_mol(mol)

        # Before generate_3d: only heavy atoms
        atoms_before = list(wrapper.inner.atoms)
        assert len(atoms_before) == 2, "Should have 2 heavy atoms initially"

        # Generate 3D with hydrogens
        wrapper.generate_3d(add_hydrogens=True, optimize=False)

        # After generate_3d: heavy atoms + hydrogens
        atoms_after = list(wrapper.inner.atoms)

        # Should have 2 C + 6 H = 8 atoms
        h_atoms = [a for a in atoms_after if a.get("atomic_num") == 1]
        print([h for h in h_atoms])

        assert len(h_atoms) == 6, f"Expected 6 H atoms, got {len(h_atoms)}"

        # All H atoms should have symbol="H"
        for h_atom in h_atoms:
            symbol = h_atom.get("symbol")
            assert symbol == "H", f"H atom has wrong symbol: {symbol}"

    def test_element_property_consistency(self):
        """Test that 'element' property is set correctly."""
        # Create molecule with different elements
        mol = Chem.MolFromSmiles("CCO")

        wrapper = RDKitWrapper.from_mol(mol)
        wrapper.generate_3d(add_hydrogens=True, optimize=False)

        atomistic = wrapper.inner
        atoms = list(atomistic.atoms)

        # Check that 'element' property matches 'symbol'
        for atom in atoms:
            symbol = atom.get("symbol")
            element = atom.get("element")

            # 'element' should be set and match 'symbol'
            # (Some code may use 'element' instead of 'symbol')
            if element is not None:
                assert (
                    element == symbol
                ), f"element={element} doesn't match symbol={symbol}"

    def test_monomer_with_ports_symbol_preservation(self):
        """Test symbol preservation when working with Monomer wrappers."""
        from molpy.core.wrappers.monomer import Monomer

        # Create simple Monomer
        monomer = Monomer()
        c1 = monomer.def_atom(symbol="C", atomic_num=6, xyz=[0.0, 0.0, 0.0])
        c2 = monomer.def_atom(symbol="C", atomic_num=6, xyz=[1.5, 0.0, 0.0])
        o1 = monomer.def_atom(symbol="O", atomic_num=8, xyz=[3.0, 0.0, 0.0])
        monomer.def_bond(c1, c2, order=1.0)
        monomer.def_bond(c2, o1, order=1.0)
        monomer.define_port("port1", o1)

        # Convert to RDKit and generate 3D
        wrapper = RDKitWrapper.from_atomistic(monomer)
        wrapper.generate_3d(add_hydrogens=True, optimize=False)

        # Get back the monomer
        final_atomistic = wrapper.inner
        atoms = list(final_atomistic.atoms)

        # Verify symbols
        heavy_atoms = [a for a in atoms if a.get("atomic_num") != 1]
        symbols = [a.get("symbol") for a in heavy_atoms]

        assert "C" in symbols, "Missing C symbol"
        assert "O" in symbols, "Missing O symbol"
        assert symbols.count("C") == 2, f"Expected 2 C atoms, got {symbols.count('C')}"
        assert symbols.count("O") == 1, f"Expected 1 O atom, got {symbols.count('O')}"


class TestRDKitWrapperAtomMapping:
    """Test atom mapping between RDKit and Atomistic."""

    def test_atom_map_after_add_hydrogens(self):
        """Test that atom mapping is correctly updated after adding H."""
        mol = Chem.MolFromSmiles("CCO")

        wrapper = RDKitWrapper.from_mol(mol)
        wrapper.generate_3d(add_hydrogens=True, optimize=False)

        # Access internal atom map
        atom_map = wrapper._atom_map

        # Verify heavy atoms are mapped
        heavy_count = 0
        for rd_idx, atom_ent in atom_map.items():
            if atom_ent.get("atomic_num") != 1:
                heavy_count += 1

        assert (
            heavy_count >= 3
        ), f"Expected at least 3 heavy atoms mapped, got {heavy_count}"

    def test_coordinate_sync_after_optimization(self):
        """Test that coordinates are correctly synced after optimization."""
        mol = Chem.MolFromSmiles("CCCC")

        wrapper = RDKitWrapper.from_mol(mol)
        wrapper.generate_3d(add_hydrogens=True, optimize=True)

        atomistic = wrapper.inner
        atoms = list(atomistic.atoms)

        # All atoms should have coordinates
        for i, atom in enumerate(atoms):
            xyz = atom.get("xyz")
            assert xyz is not None, f"Atom {i} missing coordinates"
            assert len(xyz) == 3, f"Atom {i} has invalid coordinates: {xyz}"

            # Coordinates should be non-zero (optimized structure)
            abs(xyz[0]) + abs(xyz[1]) + abs(xyz[2])
            # At least some atoms should have moved from origin
            # (We check this across all atoms below)

        # At least one atom should be away from origin
        non_zero_count = sum(
            1 for a in atoms if sum(abs(x) for x in a.get("xyz", [0, 0, 0])) > 0.1
        )
        assert non_zero_count > 0, "All atoms are at origin after optimization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
