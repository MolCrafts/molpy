"""Unit tests for molpy.external.rdkit_adapter.RDKitAdapter.

These tests focus on:
    - MP_ID-based mapping between Atomistic and RDKit Chem.Mol
    - element/symbol conventions (MolPy uses 'element', RDKit uses symbol)
    - four low-level conversion helpers:
        * _build_mol_from_atomistic
        * _build_atomistic_from_mol
        * _update_mol_from_atomistic
        * _update_atomistic_from_mol
    - sync_to_external / sync_to_internal behavior (including new atoms from RDKit)
"""

from __future__ import annotations

import pytest
from rdkit import Chem

from molpy import Atomistic
from molpy.external import RDKitAdapter, MP_ID


def _make_ethane_atomistic() -> Atomistic:
    """Create a simple ethane-like Atomistic (C-C with two H)."""
    asm = Atomistic()
    c1 = asm.def_atom(element="C")
    c2 = asm.def_atom(element="C")
    h = asm.def_atom(element="H")
    asm.def_bond(c1, c2, order=1.0)
    asm.def_bond(c1, h, order=1.0)
    return asm


class TestMPIDMapping:
    """Tests focused on MP_ID-based atom mapping."""

    def test_sync_to_external_sets_mp_id_tags(self):
        """sync_to_external should create RDKit atoms with MP_ID tags from Atomistic."""
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)

        adapter.sync_to_external()
        mol = adapter.get_external()

        assert mol.GetNumAtoms() == len(asm.atoms)

        mp_ids = []
        for rd_atom in mol.GetAtoms():
            assert rd_atom.HasProp(MP_ID)
            mp_id = rd_atom.GetIntProp(MP_ID)
            mp_ids.append(mp_id)

        # MP_IDs are unique and match Atomistic atoms
        assert len(set(mp_ids)) == len(mp_ids)
        asm_mp_ids = {int(a.get(MP_ID) or 0) for a in asm.atoms}
        assert asm_mp_ids == set(mp_ids)

    def test_sync_to_internal_adds_new_atoms_for_negative_mp_id(self):
        """RDKit atoms with negative MP_ID should become new atoms with fresh positive IDs."""
        asm = _make_ethane_atomistic()
        original_n_atoms = len(asm.atoms)
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = Chem.Mol(adapter.get_external())

        # Add a new hydrogen with negative MP_ID (mimic Generate3D behavior)
        rwmol = Chem.RWMol(mol)
        h_idx = rwmol.AddAtom(Chem.Atom("H"))
        rwmol.GetAtomWithIdx(h_idx).SetIntProp(MP_ID, -1)
        mol_with_new = rwmol.GetMol()

        adapter.set_external(mol_with_new)
        adapter.sync_to_internal()
        updated = adapter.get_internal()

        # One extra atom should be present
        assert len(updated.atoms) == original_n_atoms + 1
        # Verify no atom has None MP_ID
        for a in updated.atoms:
            assert a.get(MP_ID) is not None, f"Atom {a} has no MP_ID"
        mp_ids = {int(a.get(MP_ID)) for a in updated.atoms}
        # All MP_IDs must be non-negative (>= 0) and unique
        # Note: 0 is a valid MP_ID since IDs start from 0
        assert all(mid >= 0 for mid in mp_ids), f"Found negative MP_ID in {mp_ids}"
        assert len(mp_ids) == len(updated.atoms), f"MP_IDs not unique: {mp_ids}"


class TestElementSymbolConventions:
    """Tests for element/symbol consistency."""

    def test_build_mol_from_atomistic_uses_element_and_sets_symbol(self):
        """MolPy atoms use 'element', RDKit atoms expose symbol via GetSymbol()."""
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
        """Atoms created from RDKit must have both 'element' and 'symbol' set."""
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
    """Tests for the four low-level conversion helpers."""

    def test_build_mol_from_atomistic_round_trip(self):
        """_build_mol_from_atomistic followed by _build_atomistic_from_mol is stable."""
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)

        mol = adapter._build_mol_from_atomistic(asm)
        asm2 = adapter._build_atomistic_from_mol(mol)

        assert len(asm2.atoms) == len(asm.atoms)
        assert len(asm2.bonds) == len(asm.bonds)

    def test_update_atomistic_from_mol_preserves_existing_atoms(self):
        """_update_atomistic_from_mol should reuse existing Atomistic atom objects via MP_ID."""
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = adapter.get_external()

        original_ids = {id(a) for a in asm.atoms}

        # Slightly perturb RDKit (no topology change)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
        mol_with_conf = Chem.Mol(mol)
        mol_with_conf.AddConformer(conf, assignId=True)

        adapter.set_external(mol_with_conf)
        adapter.sync_to_internal(update_topology=False)
        updated = adapter.get_internal()

        # Existing atoms should be the same Python objects
        updated_ids = {id(a) for a in updated.atoms[: len(asm.atoms)]}
        assert original_ids == updated_ids


class TestErrorConditions:
    """Tests for key error paths (no try/except in adapter, so errors propagate)."""

    def test_missing_mp_id_on_rdkit_atom_raises(self):
        """If a RDKit atom lacks MP_ID, sync_to_internal should raise RuntimeError."""
        asm = _make_ethane_atomistic()
        adapter = RDKitAdapter(internal=asm)
        adapter.sync_to_external()
        mol = Chem.Mol(adapter.get_external())

        # Clear MP_ID from one atom
        mol.GetAtomWithIdx(0).ClearProp(MP_ID)
        adapter.set_external(mol)

        with pytest.raises(RuntimeError, match="does not have mp_id property"):
            adapter.sync_to_internal()
