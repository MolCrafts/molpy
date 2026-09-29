"""molpy.Box — the constructor and ``Style`` molpy adds over the native box."""

import numpy as np
import numpy.testing as npt
import pytest

import molpy as mp


def test_diagonal_is_promoted_to_a_matrix():
    box = mp.Box([1.0, 2.0, 3.0])
    npt.assert_allclose(box.h, np.diag([1.0, 2.0, 3.0]))
    assert box.style == mp.Box.Style.ORTHOGONAL
    npt.assert_array_equal(box.pbc, [True, True, True])


def test_full_matrix_keeps_its_tilts():
    matrix = np.array([[2.0, 1.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    box = mp.Box(matrix)
    assert box.style == mp.Box.Style.TRICLINIC
    npt.assert_allclose(box.tilts, [1.0, 0.0, 0.0])


@pytest.mark.parametrize("matrix", [None, np.zeros((3, 3))])
def test_no_cell_is_a_free_box(matrix):
    box = mp.Box(matrix)
    assert box.is_free
    assert box.style == mp.Box.Style.FREE
    assert not box.cell_defined
    npt.assert_array_equal(box.pbc, [False, False, False])


def test_origin_and_pbc_pass_through():
    box = mp.Box([2.0, 3.0, 4.0], pbc=[True, False, True], origin=[1.0, 2.0, 3.0])
    npt.assert_array_equal(box.pbc, [True, False, True])
    npt.assert_allclose(box.origin, [1.0, 2.0, 3.0])


def test_bad_shape_raises():
    with pytest.raises(ValueError, match="matrix"):
        mp.Box(np.zeros((2, 2)))


def test_is_a_native_box():
    box = mp.Box([10.0, 10.0, 10.0])
    wrapped = box.wrap(np.array([[12.0, -1.0, 5.0]]))
    npt.assert_allclose(wrapped, [[2.0, 9.0, 5.0]])
