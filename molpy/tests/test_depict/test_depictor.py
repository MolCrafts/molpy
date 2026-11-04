"""``mp.depict.Depictor``: the binding seam of the 2D layout."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp


def test_layout_returns_a_flat_copy_carrying_charges_and_ports():
    unit = mp.io.smiles.read_smiles_fragment("[<]c1ccc(C[N+](C)(C)C)cc1").to_template()
    flat = mp.depict.Depictor().layout(unit)

    assert isinstance(flat, mp.Atomistic)
    assert flat.n_atoms == unit.n_atoms and flat.n_ports == unit.n_ports
    frame = flat.to_frame()
    atoms = frame["atoms"]
    np.testing.assert_array_equal(np.asarray(atoms["z"]), 0.0)
    assert int(np.asarray(atoms["formal_charge"]).max()) == 1
    # Every bond drawn at the bond length.
    xy = np.stack([np.asarray(atoms["x"]), np.asarray(atoms["y"])], 1)
    i, j = np.asarray(frame["bonds"]["atomi"]), np.asarray(frame["bonds"]["atomj"])
    np.testing.assert_allclose(np.linalg.norm(xy[i] - xy[j], axis=1), 1.5, rtol=1e-9)
    # The input graph has no coordinates and still has none.
    assert "x" not in unit.to_frame()["atoms"].keys()


def test_bond_length_is_the_constructor_argument():
    mol = mp.io.smiles.read_smiles("CC").to_atomistic()
    frame = mp.depict.Depictor(bond_length=2.0).layout(mol).to_frame()
    x, y = np.asarray(frame["atoms"]["x"]), np.asarray(frame["atoms"]["y"])
    assert np.hypot(x[1] - x[0], y[1] - y[0]) == pytest.approx(2.0)


def test_a_non_positive_bond_length_is_a_value_error():
    with pytest.raises(ValueError, match="bond_length"):
        mp.depict.Depictor(bond_length=0.0)


def test_layout_refuses_a_non_atomistic():
    with pytest.raises(TypeError, match="Atomistic"):
        mp.depict.Depictor().layout(mp.Frame())
