"""to_rdkit / from_rdkit: pure conversion between an Atomistic and an RDKit Mol."""

import numpy as np
import pytest

pytest.importorskip("rdkit")
from rdkit import Chem
from rdkit.Chem import AllChem

import molpy as mp
from molpy.adapter import PROP_PREFIX, from_rdkit, generate_3d, to_rdkit


def _ethanol_heavy() -> mp.Atomistic:
    m = mp.Atomistic(name="ethanol")
    c1 = m.def_atom(element="C", x=0.0, y=0.0, z=0.0, name="C1", res_name="ETH")
    c2 = m.def_atom(element="C", x=1.5, y=0.0, z=0.0, name="C2", res_name="ETH")
    o = m.def_atom(element="O", x=2.2, y=1.2, z=0.0, name="O1", res_name="ETH")
    for atom, q in zip((c1, c2, o), (-0.18, 0.15, -0.68), strict=True):
        atom["charge"] = q
        atom["res_id"] = 1
        atom["type"] = "c3" if atom["element"] == "C" else "oh"
    m.def_bond(c1, c2, bond_type=1, bond_number=1, type="c3-c3")
    m.def_bond(c2, o, bond_type=1, bond_number=1, type="c3-oh")
    return m


def _snapshot(m: mp.Atomistic) -> tuple:
    return (
        [dict(a.items()) for a in m.atoms],
        [dict(b.items()) for b in m.bonds],
        sorted(m.columns()),
        dict(m.props),
    )


class TestToRdkit:
    def test_mol_mirrors_atoms_bonds_and_coordinates(self):
        m = _ethanol_heavy()
        mol = to_rdkit(m)
        assert [a.GetSymbol() for a in mol.GetAtoms()] == ["C", "C", "O"]
        assert mol.GetNumBonds() == 2
        np.testing.assert_allclose(
            mol.GetConformer().GetPositions(), m.atoms["x", "y", "z"]
        )

    def test_the_input_is_not_mutated(self):
        m = _ethanol_heavy()
        before = _snapshot(m)
        to_rdkit(m)
        assert _snapshot(m) == before

    def test_bond_classes_map_one_to_one(self):
        m = mp.Atomistic()
        a = m.def_atom(element="C")
        b = m.def_atom(element="O")
        m.def_bond(a, b, bond_type=2, bond_number=2)
        mol = to_rdkit(m)
        assert mol.GetBondWithIdx(0).GetBondType() == Chem.BondType.DOUBLE
        assert mol.GetNumConformers() == 0

    def test_molpy_components_ride_as_prefixed_props(self):
        mol = to_rdkit(_ethanol_heavy())
        atom = mol.GetAtomWithIdx(2)
        assert atom.GetProp(PROP_PREFIX + "name") == "O1"
        assert atom.GetDoubleProp(PROP_PREFIX + "charge") == pytest.approx(-0.68)
        assert mol.GetBondWithIdx(1).GetProp(PROP_PREFIX + "type") == "c3-oh"
        assert mol.GetProp(PROP_PREFIX + "name") == "ethanol"

    def test_bracket_hydrogen_count_reaches_rdkit(self):
        pyrrole = mp.io.smiles.SmilesIr("c1cc[nH]c1").to_atomistic()
        mol = to_rdkit(pyrrole)
        assert mol.GetAtomWithIdx(3).GetTotalNumHs() == 1

    def test_a_coordinate_hole_is_an_error_not_a_zero(self):
        m = _ethanol_heavy()
        m.def_atom(element="H")  # no x/y/z
        with pytest.raises(KeyError):
            to_rdkit(m)

    def test_a_non_scalar_graph_prop_is_refused_not_dropped(self):
        m = mp.Atomistic(meta={"source": "x"})
        m.def_atom(element="C")
        with pytest.raises(TypeError, match="meta"):
            to_rdkit(m)


class TestFromRdkit:
    def test_a_plain_rdkit_mol_becomes_an_atomistic(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
        m = from_rdkit(mol)
        assert list(m.atoms["element"]) == [a.GetSymbol() for a in mol.GetAtoms()]
        assert len(list(m.bonds)) == mol.GetNumBonds()
        assert "x" not in m.columns()

    def test_the_input_mol_is_not_mutated(self):
        mol = to_rdkit(_ethanol_heavy())
        before = (
            Chem.MolToMolBlock(mol),
            [sorted(a.GetPropsAsDict(False, False).items()) for a in mol.GetAtoms()],
        )
        from_rdkit(mol)
        after = (
            Chem.MolToMolBlock(mol),
            [sorted(a.GetPropsAsDict(False, False).items()) for a in mol.GetAtoms()],
        )
        assert after == before

    def test_round_trip_keeps_atom_bond_and_graph_data(self):
        m = _ethanol_heavy()
        back = from_rdkit(to_rdkit(m))
        assert _snapshot(back) == _snapshot(m)

    def test_numeric_looking_strings_stay_strings(self):
        m = mp.Atomistic()
        m.def_atom(element="C", name="1")
        assert from_rdkit(to_rdkit(m)).atoms["name"].tolist() == ["1"]

    def test_rdkit_coordinates_and_added_atoms_come_back(self):
        m = _ethanol_heavy()
        mol = Chem.AddHs(to_rdkit(m), addCoords=True)
        mol.GetConformer().SetAtomPosition(0, (9.0, 9.0, 9.0))
        back = from_rdkit(mol)
        assert len(back.entities()) == 9
        assert back.atoms["x", "y", "z"][0].tolist() == [9.0, 9.0, 9.0]
        assert back.atoms["name"].tolist()[:3] == ["C1", "C2", "O1"]
        h = list(back.atoms)[3]
        assert h["element"] == "H" and "name" not in h.keys()

    def test_atoms_rdkit_removes_are_gone_with_their_bonds(self):
        m = _ethanol_heavy()
        hydrogenated = Chem.AddHs(to_rdkit(m), addCoords=True)
        back = from_rdkit(Chem.RemoveHs(hydrogenated))
        assert list(back.atoms["element"]) == ["C", "C", "O"]
        assert len(list(back.bonds)) == 2
        assert [b["type"] for b in back.bonds] == ["c3-c3", "c3-oh"]

    def test_a_bond_type_rdkit_changes_resets_the_bond_number(self):
        benzene = mp.io.smiles.SmilesIr("c1ccccc1").to_atomistic()
        mol = Chem.RWMol(to_rdkit(benzene))
        Chem.Kekulize(mol, clearAromaticFlags=True)
        back = from_rdkit(mol)
        assert sorted(b["bond_type"] for b in back.bonds) == [1, 1, 1, 2, 2, 2]
        for bond in back.bonds:
            assert bond["bond_number"] == bond["bond_type"]

    def test_conf_id_selects_the_conformer(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=2, randomSeed=7))
        back = from_rdkit(mol, conf_id=ids[1])
        np.testing.assert_allclose(
            back.atoms["x", "y", "z"], mol.GetConformer(ids[1]).GetPositions()
        )


def test_generate_3d_returns_a_new_hydrogenated_structure_with_coordinates():
    m = _ethanol_heavy()
    before = _snapshot(m)
    out = generate_3d(m, optimize=False)
    assert len(out.entities()) == 9  # C2H5OH
    assert np.isfinite(out.atoms["x", "y", "z"]).all()
    assert out.atoms["name"].tolist()[:3] == ["C1", "C2", "O1"]
    assert _snapshot(m) == before


def test_generate_3d_embeds_the_structure_as_it_is_now():
    m = _ethanol_heavy()
    generate_3d(m, optimize=False)
    list(m.atoms)[2]["element"] = "N"
    out = generate_3d(m, optimize=False)
    assert list(out.atoms["element"])[:3] == ["C", "C", "N"]
