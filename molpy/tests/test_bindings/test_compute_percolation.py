"""Percolation of a bonded network under periodic boundaries (Livraghi et al. 2021)."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp
from molpy.compute import Percolation


def test_a_chain_closed_across_the_cell_percolates_in_one_dimension() -> None:
    positions = np.array([[0.5, 5, 5], [3.0, 5, 5], [5.5, 5, 5], [8.0, 5, 5]], dtype=float)
    out = Percolation().compute(
        positions=positions, simbox=mp.Box.cube(10.0), bonds=[[0, 1], [1, 2], [2, 3], [3, 0]]
    )
    assert out.percolation_dims == [1]
    assert out.translations == [[(1, 0, 0)]]


def test_a_periodic_net_percolates_in_three_dimensions() -> None:
    def index(x: int, y: int, z: int) -> int:
        return (z * 3 + y) * 3 + x

    positions = np.array(
        [[0.5 + 2 * x, 0.5 + 2 * y, 0.5 + 2 * z] for z in range(3) for y in range(3) for x in range(3)],
        dtype=float,
    )
    bonds = []
    for z in range(3):
        for y in range(3):
            for x in range(3):
                bonds += [
                    [index(x, y, z), index((x + 1) % 3, y, z)],
                    [index(x, y, z), index(x, (y + 1) % 3, z)],
                    [index(x, y, z), index(x, y, (z + 1) % 3)],
                ]
    out = Percolation().compute(positions=positions, simbox=mp.Box.cube(6.0), bonds=bonds)
    assert (out.n_components, out.max_dim()) == (1, 3)


def test_a_finite_molecule_is_made_whole_by_its_unwrap_images() -> None:
    positions = np.array([[9.5, 5, 5], [0.5, 5, 5]], dtype=float)
    out = Percolation().compute(positions=positions, simbox=mp.Box.cube(10.0), bonds=[[0, 1]])
    assert out.percolation_dims == [0]
    assert out.unwrap_images.tolist() == [[0, 0, 0], [1, 0, 0]]


def test_a_bond_minimum_image_cannot_read_is_refused() -> None:
    positions = np.array([[0.0, 5, 5], [5.0, 5, 5]], dtype=float)
    with pytest.raises(ValueError, match="half"):
        Percolation().compute(positions=positions, simbox=mp.Box.cube(10.0), bonds=[[0, 1]])
