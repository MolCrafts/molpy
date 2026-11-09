"""FFI seam of ``Atomistic.coords`` / ``formula`` and
``Element.fill_masses``: dtypes and shapes in and out, refusals as
``ValueError`` / ``KeyError``. The core proves the values."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp


def _atoms(*numbers: int) -> mp.Atomistic:
    mol = mp.Atomistic()
    for z in numbers:
        mol.add_atom(atomic_number=z)
    return mol


def _water() -> mp.Atomistic:
    return _atoms(8, 1, 1)


def _with_coords(mol: mp.Atomistic) -> mp.Atomistic:
    mol.coords = np.arange(mol.n_atoms * 3, dtype=float).reshape(-1, 3)
    return mol


def test_coords_round_trip_in_atom_row_order() -> None:
    mol = _with_coords(_water())
    points = mol.coords
    assert points.dtype == np.float64 and points.shape == (mol.n_atoms, 3)
    assert points.tolist() == np.arange(mol.n_atoms * 3).reshape(-1, 3).tolist()
    assert mol.atoms[1]["y"] == 4.0


def test_coords_of_the_wrong_shape_are_refused() -> None:
    mol = _with_coords(_water())
    with pytest.raises(ValueError):
        mol.coords = np.zeros((mol.n_atoms + 1, 3))
    assert mol.atoms[0]["x"] == 0.0


def test_coords_without_positions_are_refused() -> None:
    with pytest.raises(ValueError):
        _ = _water().coords


def test_formula_is_in_hill_order() -> None:
    assert _atoms(8, 6, 1, 1, 6, 1, 1, 1, 1).formula() == "C2H6O"
    assert _water().formula() == "H2O"


def test_fill_masses_writes_the_mass_column() -> None:
    frame = _water().to_frame()
    mp.Element.fill_masses(frame)
    masses = frame["atoms"]["mass"]
    assert masses.dtype == np.float64
    oxygen = mp.Element.by_symbol("O").mass
    assert sorted(masses.tolist())[-1] == pytest.approx(oxygen)
