"""RDKitAdapter: the ``mp_id`` join between an Atomistic and an RDKit Mol."""

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem

import molpy as mp
from molpy.adapter.rdkit import MP_ID, RDKitAdapter


def _ethanol_heavy() -> mp.Atomistic:
    m = mp.Atomistic()
    c1 = m.def_atom(element="C", x=0.0, y=0.0, z=0.0)
    c2 = m.def_atom(element="C", x=1.5, y=0.0, z=0.0)
    o = m.def_atom(element="O", x=2.2, y=1.2, z=0.0)
    m.def_bond(c1, c2, bond_type=1, bond_number=1)
    m.def_bond(c2, o, bond_type=1, bond_number=1)
    return m


def _tagged(mol: Chem.Mol) -> Chem.Mol:
    for idx, rd_atom in enumerate(mol.GetAtoms()):
        rd_atom.SetIntProp(MP_ID, idx)
    return mol


class TestTags:
    def test_every_atom_gets_a_unique_tag(self):
        m = _ethanol_heavy()
        RDKitAdapter(internal=m)
        assert sorted(m.column(MP_ID).tolist()) == [0, 1, 2]

    def test_existing_tags_are_kept_and_holes_filled_above_them(self):
        m = _ethanol_heavy()
        first = next(iter(m.atoms))
        first[MP_ID] = 7
        RDKitAdapter(internal=m)
        tags = m.column(MP_ID).tolist()
        assert tags[0] == 7
        assert sorted(tags) == [7, 8, 9]

    def test_duplicate_tags_raise(self):
        m = _ethanol_heavy()
        for atom in m.atoms:
            atom[MP_ID] = 1
        with pytest.raises(ValueError, match="duplicate"):
            RDKitAdapter(internal=m)


class TestSyncToExternal:
    def test_mol_mirrors_atoms_bonds_tags_and_coordinates(self):
        m = _ethanol_heavy()
        adapter = RDKitAdapter(internal=m)
        adapter.sync_to_external()
        mol = adapter.mol
        assert [a.GetSymbol() for a in mol.GetAtoms()] == ["C", "C", "O"]
        assert mol.GetNumBonds() == 2
        assert [a.GetIntProp(MP_ID) for a in mol.GetAtoms()] == m.column(MP_ID).tolist()
        np.testing.assert_allclose(
            mol.GetConformer().GetPositions(), m.atoms["x", "y", "z"]
        )

    def test_bond_classes_map_one_to_one(self):
        m = mp.Atomistic()
        a = m.def_atom(element="C")
        b = m.def_atom(element="O")
        m.def_bond(a, b, bond_type=2, bond_number=2)
        adapter = RDKitAdapter(internal=m)
        adapter.sync_to_external()
        assert adapter.mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE
        assert adapter.mol.GetNumConformers() == 0

    def test_a_coordinate_hole_is_an_error_not_a_zero(self):
        m = _ethanol_heavy()
        m.def_atom(element="H")  # no x/y/z
        with pytest.raises(KeyError):
            RDKitAdapter(internal=m).sync_to_external()


class TestSyncToInternal:
    def test_fresh_internal_is_built_from_the_mol(self):
        mol = _tagged(Chem.AddHs(Chem.MolFromSmiles("CO")))
        adapter = RDKitAdapter(external=mol)
        adapter.sync_to_internal()
        m = adapter.internal
        assert list(m.atoms["element"]) == [a.GetSymbol() for a in mol.GetAtoms()]
        assert len(list(m.bonds)) == mol.GetNumBonds()
        assert m.column(MP_ID).tolist() == list(range(mol.GetNumAtoms()))

    def test_negative_tags_become_new_atoms_and_bonds_are_rebuilt(self):
        m = _ethanol_heavy()
        adapter = RDKitAdapter(internal=m)
        adapter.sync_to_external()
        mol = Chem.AddHs(Chem.Mol(adapter.mol), addCoords=True)
        for rd_atom in mol.GetAtoms():
            if not rd_atom.HasProp(MP_ID):
                rd_atom.SetIntProp(MP_ID, -1)

        adapter.set_external(mol)
        adapter.sync_to_internal()

        assert len(m.entities()) == mol.GetNumAtoms()
        tags = m.column(MP_ID).tolist()
        assert len(set(tags)) == len(tags)
        assert [a.GetIntProp(MP_ID) for a in mol.GetAtoms()] == tags
        assert len(list(m.bonds)) == mol.GetNumBonds()
        assert m.atoms["x", "y", "z"].shape == (mol.GetNumAtoms(), 3)

    def test_known_atoms_are_updated_in_place(self):
        m = _ethanol_heavy()
        adapter = RDKitAdapter(internal=m)
        adapter.sync_to_external()
        mol = Chem.Mol(adapter.mol)
        mol.GetConformer().SetAtomPosition(0, (9.0, 9.0, 9.0))
        adapter.set_external(mol)
        adapter.sync_to_internal()
        assert len(m.entities()) == 3
        assert m.atoms["x", "y", "z"][0].tolist() == [9.0, 9.0, 9.0]

    def test_an_untagged_rdkit_atom_is_an_error(self):
        adapter = RDKitAdapter(
            internal=_ethanol_heavy(), external=Chem.MolFromSmiles("CCO")
        )
        with pytest.raises(RuntimeError, match=MP_ID):
            adapter.sync_to_internal()


def test_generate_3d_returns_a_new_hydrogenated_structure_with_coordinates():
    m = _ethanol_heavy()
    adapter = RDKitAdapter(internal=m)
    out = adapter.generate_3d(optimize=False)
    assert out is not m
    assert len(out.entities()) == 9  # C2H5OH
    assert out.atoms["x", "y", "z"].shape == (9, 3)
    assert np.isfinite(out.atoms["x", "y", "z"]).all()
    assert len(m.entities()) == 3  # the adapter's own structure is untouched
