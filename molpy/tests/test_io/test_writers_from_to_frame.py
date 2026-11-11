"""Symbol-spelling writers take a frame straight from ``Atomistic.to_frame()``.

An ion parsed from SMILES and embedded states its formal charge on one atom
and draws its hydrogens; the frame it exports must still write as PDB, XYZ
and (typed) MOL2 with no column the caller has to add.
"""

from __future__ import annotations

import molpy as mp


def _hydroxide() -> mp.Atomistic:
    mol = mp.io.smiles.read_smiles("[OH-]").to_atomistic()
    mol, _ = mp.conformer.Conformer(seed=1).generate(mol)
    return mol


def test_pdb_and_xyz_write_an_embedded_ion():
    frame = _hydroxide().to_frame()
    pdb, xyz = bytearray(), bytearray()
    mp.io.write_pdb(pdb, frame)
    mp.io.write_xyz(xyz, frame)
    assert mp.io.read_pdb(bytes(pdb))["atoms"].n_rows == 2
    lines = bytes(xyz).decode().splitlines()
    assert lines[0].strip() == "2"
    assert sorted(line.split()[0] for line in lines[2:4]) == ["H", "O"]


def test_mol2_names_atoms_from_their_elements():
    typed = mp.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(_hydroxide())
    out = bytearray()
    mp.io.write_mol2(out, typed.to_frame())
    names = list(mp.io.read_mol2(bytes(out))["atoms"]["name"])
    assert names == ["O1", "H2"]


def test_charges_and_aromaticity_from_smiles_carry_no_validity_mask():
    """Every atom's formal charge and aromaticity is known (absent is 0 /
    not aromatic), so the exported columns are dense."""
    atoms = (
        mp.io.smiles.read_smiles("c1ccccc1C[N+](C)(C)C")
        .to_atomistic()
        .to_frame()["atoms"]
    )
    assert atoms.validity("formal_charge") is None
    assert atoms.validity("is_aromatic") is None
    assert int(atoms["formal_charge"].sum()) == 1
    assert int(atoms["is_aromatic"].sum()) == 6


def test_an_embedded_molecule_states_every_hydrogen_count():
    mol = mp.io.smiles.read_smiles("C[N+](C)(C)C").to_atomistic()
    mol, _ = mp.conformer.Conformer(seed=1).generate(mol)
    atoms = mol.to_frame()["atoms"]
    assert atoms.validity("h_count") is None
    assert int(atoms["h_count"].max()) == 0


def test_xyz_round_trips_formal_charge():
    frame = _hydroxide().to_frame()
    out = bytearray()
    mp.io.write_xyz(out, frame)
    back = mp.io.read_xyz(bytes(out))["atoms"]
    assert sorted(int(q) for q in back["formal_charge"]) == [-1, 0]
