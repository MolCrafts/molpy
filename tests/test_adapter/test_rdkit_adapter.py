"""Unit tests for molpy.adapter.rdkit.RDKitAdapter.

These tests focus on:
    - MP_ID-based mapping between Atomistic and RDKit Chem.Mol
    - element/symbol conventions
    - conversion helpers and sync behavior
"""

from __future__ import annotations

import pytest

# Skip entire test module if RDKit is not available
pytest.importorskip("rdkit", reason="RDKit is not installed")

from rdkit import Chem

from molpy import Atomistic
from molpy.adapter import MP_ID, RDKitAdapter

# Type assertions: these are guaranteed to be non-None after importorskip
assert MP_ID is not None
assert RDKitAdapter is not None


def _make_ethane_atomistic() -> Atomistic:
    """Create a simple ethane-like Atomistic (C-C with one H)."""

    asm = Atomistic()
    c1 = asm.def_atom(element="C")
    c2 = asm.def_atom(element="C")
    h = asm.def_atom(element="H")
    asm.def_bond(c1, c2, order=1.0)
    asm.def_bond(c1, h, order=1.0)
    return asm


class TestMPIDMapping:
    def test_sync_to_external_sets_mp_id_tags(self):
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)

        adapter.sync_to_external()
        mol = adapter.get_external()

        assert mol.GetNumAtoms() == len(asm.atoms)

        mp_ids = []
        for rd_atom in mol.GetAtoms():
            assert rd_atom.HasProp(MP_ID)
            mp_ids.append(rd_atom.GetIntProp(MP_ID))

        assert len(set(mp_ids)) == len(mp_ids)
        asm_mp_ids = {int(a.get(MP_ID) or 0) for a in asm.atoms}
        assert asm_mp_ids == set(mp_ids)

    def test_sync_to_internal_adds_new_atoms_for_negative_mp_id(self):
        asm = _make_ethane_atomistic()
        original_n_atoms = len(asm.atoms)
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = Chem.Mol(adapter.get_external())

        rwmol = Chem.RWMol(mol)
        h_idx = rwmol.AddAtom(Chem.Atom("H"))
        rwmol.GetAtomWithIdx(h_idx).SetIntProp(MP_ID, -1)
        mol_with_new = rwmol.GetMol()

        adapter.set_external(mol_with_new)
        adapter.sync_to_internal()
        updated = adapter.get_internal()

        assert len(updated.atoms) == original_n_atoms + 1
        for a in updated.atoms:
            assert a.get(MP_ID) is not None
        mp_ids = {int(a.get(MP_ID)) for a in updated.atoms}
        assert all(mid >= 0 for mid in mp_ids)
        assert len(mp_ids) == len(updated.atoms)


class TestElementSymbolConventions:
    def test_build_mol_from_atomistic_uses_element(self):
        asm = Atomistic()
        c = asm.def_atom(element="C")
        h = asm.def_atom(element="H")
        asm.def_bond(c, h, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        assert sorted(symbols) == ["C", "H"]

    def test_build_atomistic_from_mol_sets_element_and_symbol(self):
        mol = Chem.RWMol()
        c = Chem.Atom("C")
        c.SetIntProp(MP_ID, 1)
        h = Chem.Atom("H")
        h.SetIntProp(MP_ID, 2)
        i_c = mol.AddAtom(c)
        i_h = mol.AddAtom(h)
        mol.AddBond(i_c, i_h, Chem.BondType.SINGLE)
        final = mol.GetMol()

        adapter = RDKitAdapter(external=final)
        adapter.sync_to_internal()
        asm = adapter.get_internal()

        assert len(asm.atoms) == 2
        for a in asm.atoms:
            assert a.get("element") in {"C", "H"}
            assert a.get("symbol") in {"C", "H"}


class TestConversionHelpers:
    def test_build_mol_from_atomistic_round_trip(self):
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)

        mol = adapter._build_mol_from_atomistic(asm)
        asm2 = adapter._build_atomistic_from_mol(mol)

        assert len(asm2.atoms) == len(asm.atoms)
        assert len(asm2.bonds) == len(asm.bonds)

    def test_update_atomistic_from_mol_preserves_existing_atoms(self):
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        original_ids = {id(a) for a in asm.atoms}

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        mol_with_conf = Chem.Mol(mol)
        mol_with_conf.AddConformer(conf, assignId=True)

        adapter.set_external(mol_with_conf)
        adapter.sync_to_internal(update_topology=False)
        updated = adapter.get_internal()

        updated_ids = {id(a) for a in updated.atoms[: len(asm.atoms)]}
        assert original_ids == updated_ids


class TestErrorConditions:
    def test_missing_mp_id_on_rdkit_atom_raises(self):
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = Chem.Mol(adapter.get_external())

        mol.GetAtomWithIdx(0).ClearProp(MP_ID)
        adapter.set_external(mol)

        with pytest.raises(RuntimeError, match="does not have mp_id property"):
            adapter.sync_to_internal()


class TestCoordinateHandling:
    """Tests for coordinate synchronization between Atomistic and RDKit."""

    def test_atomistic_with_coordinates_syncs_to_mol(self):
        """Test that coordinates are properly transferred to RDKit conformer."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C", x=1.0, y=2.0, z=3.0)
        c2 = asm.def_atom(element="C", x=4.0, y=5.0, z=6.0)
        h = asm.def_atom(element="H", x=7.0, y=8.0, z=9.0)
        asm.def_bond(c1, c2, order=1.0)
        asm.def_bond(c1, h, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        assert mol.GetNumConformers() == 1
        conf = mol.GetConformer()
        for rd_atom in mol.GetAtoms():
            idx = rd_atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            # Find corresponding atom in asm
            mp_id = rd_atom.GetIntProp(MP_ID)
            atom = next(
                a
                for a in asm.atoms
                if a.get(MP_ID) is not None and int(a.get(MP_ID)) == mp_id
            )
            atom_x = atom.get("x")
            atom_y = atom.get("y")
            atom_z = atom.get("z")
            assert atom_x is not None and atom_y is not None and atom_z is not None
            assert abs(pos.x - float(atom_x)) < 1e-6
            assert abs(pos.y - float(atom_y)) < 1e-6
            assert abs(pos.z - float(atom_z)) < 1e-6

    def test_mol_with_conformer_syncs_to_atomistic(self):
        """Test that RDKit conformer coordinates are transferred to Atomistic."""
        mol = Chem.RWMol()
        c1 = Chem.Atom("C")
        c1.SetIntProp(MP_ID, 1)
        c2 = Chem.Atom("C")
        c2.SetIntProp(MP_ID, 2)
        idx_c1 = mol.AddAtom(c1)
        idx_c2 = mol.AddAtom(c2)
        mol.AddBond(idx_c1, idx_c2, Chem.BondType.SINGLE)
        final_mol = mol.GetMol()

        conf = Chem.Conformer(2)
        conf.SetAtomPosition(0, (1.0, 2.0, 3.0))
        conf.SetAtomPosition(1, (4.0, 5.0, 6.0))
        final_mol.AddConformer(conf, assignId=True)

        adapter = RDKitAdapter(external=final_mol)
        adapter.sync_to_internal()
        asm = adapter.get_internal()

        assert len(asm.atoms) == 2
        atoms_by_id = {
            int(a.get(MP_ID)): a for a in asm.atoms if a.get(MP_ID) is not None
        }
        atom1 = atoms_by_id[1]
        atom2 = atoms_by_id[2]
        assert (
            atom1.get("x") is not None
            and atom1.get("y") is not None
            and atom1.get("z") is not None
        )
        assert (
            atom2.get("x") is not None
            and atom2.get("y") is not None
            and atom2.get("z") is not None
        )
        assert abs(float(atom1.get("x")) - 1.0) < 1e-6
        assert abs(float(atom1.get("y")) - 2.0) < 1e-6
        assert abs(float(atom1.get("z")) - 3.0) < 1e-6
        assert abs(float(atom2.get("x")) - 4.0) < 1e-6
        assert abs(float(atom2.get("y")) - 5.0) < 1e-6
        assert abs(float(atom2.get("z")) - 6.0) < 1e-6

    def test_incomplete_coordinates_raises_error(self):
        """Test that incomplete coordinates raise an error."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C", x=1.0, y=2.0, z=3.0)
        c2 = asm.def_atom(element="C", x=4.0)  # Missing y and z
        asm.def_bond(c1, c2, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        with pytest.raises(ValueError, match="incomplete coordinates"):
            adapter.sync_to_external()


class TestBondTypes:
    """Tests for different bond types."""

    def test_single_bond(self):
        """Test single bond conversion."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        bond = mol.GetBondBetweenAtoms(0, 1)
        assert bond.GetBondType() == Chem.BondType.SINGLE

    def test_double_bond(self):
        """Test double bond conversion."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=2.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        bond = mol.GetBondBetweenAtoms(0, 1)
        assert bond.GetBondType() == Chem.BondType.DOUBLE

    def test_triple_bond(self):
        """Test triple bond conversion."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=3.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        bond = mol.GetBondBetweenAtoms(0, 1)
        assert bond.GetBondType() == Chem.BondType.TRIPLE

    def test_aromatic_bond(self):
        """Test aromatic bond conversion."""
        # Create a benzene ring (aromatic 6-membered ring)
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        c3 = asm.def_atom(element="C")
        c4 = asm.def_atom(element="C")
        c5 = asm.def_atom(element="C")
        c6 = asm.def_atom(element="C")
        # Create aromatic ring with alternating single/double bonds
        # In RDKit, aromatic bonds in rings are valid
        asm.def_bond(c1, c2, order=1.5)  # Aromatic
        asm.def_bond(c2, c3, order=1.5)  # Aromatic
        asm.def_bond(c3, c4, order=1.5)  # Aromatic
        asm.def_bond(c4, c5, order=1.5)  # Aromatic
        asm.def_bond(c5, c6, order=1.5)  # Aromatic
        asm.def_bond(c6, c1, order=1.5)  # Aromatic (closes ring)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        # Check that bonds in the ring are aromatic
        bond = mol.GetBondBetweenAtoms(0, 1)
        assert bond.GetBondType() == Chem.BondType.AROMATIC

    def test_bond_round_trip(self):
        """Test bond order preservation through round trip."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=2.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        # Create new adapter from mol
        adapter2 = RDKitAdapter(external=mol)
        adapter2.sync_to_internal()
        asm2 = adapter2.get_internal()

        assert len(asm2.bonds) == 1
        bond_order = asm2.bonds[0].get("order")
        assert bond_order is not None
        assert abs(float(bond_order) - 2.0) < 1e-6


class TestFormalCharge:
    """Tests for formal charge handling."""

    def test_atomistic_with_charge_syncs_to_mol(self):
        """Test that formal charges are transferred to RDKit."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C", charge=1)
        c2 = asm.def_atom(element="C", charge=-1)
        asm.def_bond(c1, c2, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        atoms_by_mp_id = {}
        for rd_atom in mol.GetAtoms():
            mp_id = rd_atom.GetIntProp(MP_ID)
            atoms_by_mp_id[mp_id] = rd_atom

        # Find atoms with charges
        charged_atoms = [a for a in asm.atoms if a.get("charge") is not None]
        for atom in charged_atoms:
            mp_id = int(atom.get(MP_ID))
            rd_atom = atoms_by_mp_id[mp_id]
            assert rd_atom.GetFormalCharge() == int(atom.get("charge"))

    def test_mol_with_charge_syncs_to_atomistic(self):
        """Test that RDKit formal charges are transferred to Atomistic."""
        mol = Chem.RWMol()
        c1 = Chem.Atom("C")
        c1.SetIntProp(MP_ID, 1)
        c1.SetFormalCharge(1)
        c2 = Chem.Atom("C")
        c2.SetIntProp(MP_ID, 2)
        c2.SetFormalCharge(-1)
        mol.AddAtom(c1)
        mol.AddAtom(c2)
        mol.AddBond(0, 1, Chem.BondType.SINGLE)
        final_mol = mol.GetMol()

        adapter = RDKitAdapter(external=final_mol)
        adapter.sync_to_internal()
        asm = adapter.get_internal()

        atoms_by_id = {
            int(a.get(MP_ID)): a for a in asm.atoms if a.get(MP_ID) is not None
        }
        assert atoms_by_id[1].get("charge") == 1
        assert atoms_by_id[2].get("charge") == -1


class TestUpdateBehavior:
    """Tests for update_atomistic_from_mol behavior."""

    def test_update_topology_true_rebuilds_bonds(self):
        """Test that update_topology=True rebuilds all bonds."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=1.0)
        original_bonds = list(asm.bonds)

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        # Add a new atom and bond in RDKit
        rwmol = Chem.RWMol(mol)
        h = Chem.Atom("H")
        h.SetIntProp(MP_ID, 999)
        h_idx = rwmol.AddAtom(h)
        # Add bond to first carbon
        c1_idx = 0  # Assuming first atom is c1
        rwmol.AddBond(c1_idx, h_idx, Chem.BondType.SINGLE)
        mol_updated = rwmol.GetMol()

        adapter.set_external(mol_updated)
        adapter.sync_to_internal(update_topology=True)
        updated = adapter.get_internal()

        assert len(updated.bonds) == 2  # Original C-C + new C-H
        bond_orders = {
            b.get("order") for b in updated.bonds if b.get("order") is not None
        }
        assert 1.0 in bond_orders

    def test_update_topology_false_preserves_bonds(self):
        """Test that update_topology=False preserves existing topology."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        asm.def_bond(c1, c2, order=1.0)
        original_bond_ids = {id(b) for b in asm.bonds}

        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        # Modify coordinates only
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        mol_with_conf = Chem.Mol(mol)
        mol_with_conf.AddConformer(conf, assignId=True)

        adapter.set_external(mol_with_conf)
        adapter.sync_to_internal(update_topology=False)
        updated = adapter.get_internal()

        # Bonds should be preserved
        updated_bond_ids = {id(b) for b in updated.bonds}
        assert original_bond_ids == updated_bond_ids


class TestAtomIDAssignment:
    """Tests for automatic atom ID assignment."""

    def test_atoms_without_ids_get_assigned(self):
        """Test that atoms without IDs get automatically assigned."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        # Don't set IDs
        asm.def_bond(c1, c2, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        # _ensure_atom_ids should be called in __init__
        for atom in asm.atoms:
            assert atom.get("id") is not None
            assert atom.get(MP_ID) is not None

    def test_atoms_with_existing_ids_are_preserved(self):
        """Test that existing IDs are preserved."""
        asm = Atomistic()
        c1 = asm.def_atom(element="C")
        c2 = asm.def_atom(element="C")
        c1["id"] = 100
        c2["id"] = 200
        asm.def_bond(c1, c2, order=1.0)

        adapter = RDKitAdapter(internal=asm)
        assert c1.get("id") == 100
        assert c2.get("id") == 200
        assert c1.get(MP_ID) is not None and int(c1.get(MP_ID)) == 100
        assert c2.get(MP_ID) is not None and int(c2.get(MP_ID)) == 200
