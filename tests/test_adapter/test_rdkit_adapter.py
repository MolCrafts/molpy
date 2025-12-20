"""Unit tests for molpy.adapter.rdkit.RDKitAdapter.

These tests focus on:
    - MP_ID-based mapping between Atomistic and RDKit Chem.Mol
    - element/symbol conventions
    - conversion helpers and sync behavior
"""

from __future__ import annotations

import pytest

from rdkit import Chem

from molpy import Atomistic
from molpy.adapter import MP_ID, RDKitAdapter


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
