import pickle

import numpy as np
import pytest
from molpy.core import Box, Cell


class TestBoxFactories:
    """The factories are the matching Cell variant, nothing more."""

    def test_cube_is_three_equal_lengths(self):
        assert Box.cube(10.0).approx_eq(Box(Cell.lengths([10.0] * 3)), 1e-12)

    def test_ortho_is_cell_lengths(self):
        assert Box.ortho([2.0, 3.0, 4.0]).approx_eq(
            Box(Cell.lengths([2.0, 3.0, 4.0])), 1e-12
        )

    def test_from_lengths_tilts_is_cell_lengths_tilts(self):
        lengths, tilts = [10.0, 12.0, 15.0], [1.0, 0.5, 0.2]
        b = Box.from_lengths_tilts(lengths, tilts)
        assert b.approx_eq(Box(Cell.lengths_tilts(lengths, tilts)), 1e-12)
        assert b.style == "triclinic"

    def test_from_lengths_angles_is_cell_lengths_angles(self):
        lengths, angles = [10.0, 11.0, 12.0], [80.0, 85.0, 95.0]
        assert Box.from_lengths_angles(lengths, angles).approx_eq(
            Box(Cell.lengths_angles(lengths, angles)), 1e-12
        )

    def test_factories_take_origin_and_pbc(self):
        b = Box.cube(20.0, origin=[1.0, 2.0, 3.0], pbc=[True, True, False])
        assert list(b.origin) == [1.0, 2.0, 3.0]
        assert list(b.pbc) == [True, True, False]

    def test_constructor_takes_the_matrix(self):
        h = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
        assert Box(h).approx_eq(Box(Cell.matrix(h)), 1e-12)

    def test_factories_check_like_the_cell(self):
        with pytest.raises(ValueError):
            Box.cube(0.0)
        with pytest.raises(ValueError):
            Box.ortho([1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="singular"):
            Box(np.zeros((3, 3)))


class TestBoxConstruction:
    def test_cube(self):
        b = Box(Cell.lengths([10.0, 10.0, 10.0]))
        assert pytest.approx(b.volume(), abs=1e-3) == 1000.0

    def test_ortho(self):
        b = Box(Cell.lengths(np.array([2.0, 3.0, 4.0], dtype=np.float64)))
        assert pytest.approx(b.volume(), abs=1e-3) == 24.0

    def test_triclinic(self):
        h = np.array(
            [[2.0, 1.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]], dtype=np.float64
        )
        b = Box(Cell.matrix(h))
        assert b.volume() > 0
        assert b.style == "triclinic"

    def test_singular_matrix_raises_value_error(self):
        h = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="singular"):
            Box(Cell.matrix(h))

    def test_a_zero_matrix_is_singular(self):
        with pytest.raises(ValueError, match="singular"):
            Box(Cell.matrix(np.zeros((3, 3))))

    def test_an_undefined_cell_is_a_free_box(self):
        b = Box(Cell.undefined(), pbc=(False, False, False))
        assert not b.is_cell_defined
        assert b.style == "free"
        assert b.is_free
        assert list(b.pbc) == [False, False, False]
        assert b.volume() == 0.0
        np.testing.assert_array_equal(b.matrix, np.eye(3))

    def test_an_undefined_cell_may_still_say_which_axes_wrap(self):
        b = Box(Cell.undefined(), pbc=np.array([True, False, False]))
        assert not b.is_cell_defined
        assert list(b.pbc) == [True, False, False]

    def test_the_defaults_are_the_origin_and_full_periodicity(self):
        b = Box(Cell.lengths([10.0, 20.0, 30.0]))
        assert b.style == "orthogonal"
        assert b.is_cell_defined
        assert list(b.pbc) == [True, True, True]
        np.testing.assert_array_equal(b.origin, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(b.matrix, np.diag([10.0, 20.0, 30.0]))
        assert pytest.approx(b.volume()) == 6000.0

    def test_a_nested_list_is_a_matrix(self):
        b = Box(Cell.matrix([[10, 0, 0], [0, 10, 0], [0, 0, 10]]))
        assert pytest.approx(b.volume()) == 1000.0

    def test_lengths_angles_of_a_right_cell_is_orthogonal(self):
        b = Box(Cell.lengths_angles([3.0, 4.0, 5.0], [90.0, 90.0, 90.0]))
        np.testing.assert_allclose(b.matrix, np.diag([3.0, 4.0, 5.0]), atol=1e-12)
        np.testing.assert_allclose(b.angles, [90.0, 90.0, 90.0])

    def test_lengths_angles_reproduces_the_angles(self):
        b = Box(Cell.lengths_angles([3.0, 4.0, 5.0], [80.0, 95.0, 110.0]))
        np.testing.assert_allclose(b.lengths, [3.0, 4.0, 5.0])
        np.testing.assert_allclose(b.angles, [80.0, 95.0, 110.0])
        # Restricted triclinic: a along x, b in the xy plane.
        assert b.matrix[1, 0] == 0.0 and b.matrix[2, 0] == 0.0
        assert b.matrix[2, 1] == 0.0

    def test_impossible_angles_raise(self):
        with pytest.raises(ValueError, match="lengths/angles"):
            Box(Cell.lengths_angles([1.0, 1.0, 1.0], [0.0, 90.0, 90.0]))

    def test_lengths_tilts_is_the_lammps_form(self):
        b = Box(Cell.lengths_tilts([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(
            b.matrix, [[10.0, 1.0, 2.0], [0.0, 20.0, 3.0], [0.0, 0.0, 30.0]]
        )
        np.testing.assert_array_equal(b.tilts, [1.0, 2.0, 3.0])

    def test_cube_zero_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            Box(Cell.lengths([0.0, 0.0, 0.0]))

    def test_cube_negative_raises_value_error(self):
        with pytest.raises(ValueError, match="positive"):
            Box(Cell.lengths([-5.0, -5.0, -5.0]))

    def test_custom_origin(self):
        b = Box(
            Cell.lengths([10.0, 10.0, 10.0]),
            origin=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        np.testing.assert_allclose(b.origin, [1.0, 2.0, 3.0], atol=1e-6)

    def test_custom_pbc(self):
        b = Box(Cell.lengths([10.0, 10.0, 10.0]), pbc=np.array([True, False, True]))
        assert b.pbc.tolist() == [True, False, True]

    def test_the_cell_is_required(self):
        with pytest.raises(TypeError):
            Box()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            Box([10.0, 10.0, 10.0])  # lengths go through Box.ortho


class TestCell:
    def test_cells_compare_by_value(self):
        assert Cell.lengths([1.0, 2.0, 3.0]) == Cell.lengths([1.0, 2.0, 3.0])
        assert Cell.lengths([1.0, 2.0, 3.0]) != Cell.lengths([1.0, 2.0, 4.0])
        assert Cell.undefined() == Cell.undefined()
        assert Cell.undefined() != Cell.lengths([1.0, 1.0, 1.0])

    @pytest.mark.parametrize(
        "cell",
        [
            Cell.matrix(np.eye(3) * 2.0),
            Cell.lengths([1.0, 2.0, 3.0]),
            Cell.lengths_angles([1.0, 2.0, 3.0], [90.0, 90.0, 100.0]),
            Cell.lengths_tilts([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]),
            Cell.undefined(),
        ],
    )
    def test_every_variant_pickles(self, cell):
        assert pickle.loads(pickle.dumps(cell)) == cell

    def test_a_cell_is_frozen(self):
        with pytest.raises(AttributeError):
            Cell.undefined().x = 1  # type: ignore[attr-defined]


class TestBoxPickle:
    def test_a_box_round_trips(self, cubic_box):
        back = pickle.loads(pickle.dumps(cubic_box))
        assert back.approx_eq(cubic_box, 0.0)

    def test_a_triclinic_box_with_origin_and_pbc_round_trips(self):
        box = Box(
            Cell.lengths_tilts([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]),
            origin=(1.0, -2.0, 3.0),
            pbc=(True, False, True),
        )
        back = pickle.loads(pickle.dumps(box))
        assert back.approx_eq(box, 0.0)

    def test_an_undefined_box_round_trips(self):
        box = Box(Cell.undefined(), pbc=(False, False, False))
        back = pickle.loads(pickle.dumps(box))
        assert not back.is_cell_defined
        assert back.approx_eq(box, 0.0)


class TestBoxProperties:
    def test_matrix(self, cubic_box):
        h = cubic_box.matrix
        assert h.shape == (3, 3)
        assert pytest.approx(h[0, 0]) == 10.0
        assert pytest.approx(h[1, 1]) == 10.0
        assert pytest.approx(h[2, 2]) == 10.0

    def test_matrix_rebuilds_the_box(self):
        box = Box(
            Cell.lengths_angles([3.0, 4.0, 5.0], [80.0, 95.0, 110.0]),
            origin=(1.0, 2.0, 3.0),
            pbc=(True, True, False),
        )
        again = Box(Cell.matrix(box.matrix), box.origin, box.pbc)
        assert again.approx_eq(box, 0.0)

    def test_origin(self, cubic_box):
        o = cubic_box.origin
        assert o.shape == (3,)
        np.testing.assert_allclose(o, [0.0, 0.0, 0.0], atol=1e-6)

    def test_pbc(self, cubic_box):
        pbc = cubic_box.pbc
        assert pbc.shape == (3,)
        assert all(pbc)

    def test_volume(self, ortho_box):
        assert pytest.approx(ortho_box.volume(), abs=1e-2) == 750.0

    def test_lengths(self, ortho_box):
        lengths = ortho_box.lengths
        np.testing.assert_allclose(lengths, [5.0, 10.0, 15.0], atol=1e-4)

    def test_lattice_vectors(self, cubic_box):
        for i in range(3):
            v = cubic_box.lattice(i)
            assert v.shape == (3,)

    def test_inverse(self, ortho_box):
        np.testing.assert_allclose(ortho_box.inverse @ ortho_box.matrix, np.eye(3))

    def test_restricted_matrix_rotates_a_onto_x(self):
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        h = rot @ np.diag([3.0, 4.0, 5.0])
        np.testing.assert_allclose(
            Box.restricted_matrix(h), np.diag([3.0, 4.0, 5.0]), atol=1e-12
        )


class TestBoxCoordinateTransforms:
    def test_roundtrip_frac_cart(self, cubic_box, sample_points):
        frac = cubic_box.to_frac(sample_points)
        cart = cubic_box.to_cart(frac)
        np.testing.assert_allclose(cart, sample_points, atol=1e-4)

    def test_frac_values(self, cubic_box):
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        frac = cubic_box.to_frac(pts)
        np.testing.assert_allclose(frac, [[0.5, 0.5, 0.5]], atol=1e-4)

    def test_wrap(self, cubic_box):
        pts = np.array([[11.0, -1.0, 25.0]], dtype=np.float64)
        wrapped, shifts = cubic_box.wrap(pts)
        frac = cubic_box.to_frac(wrapped)
        assert np.all(frac >= 0.0)
        assert np.all(frac < 1.0)
        np.testing.assert_array_equal(shifts, [[1, -1, 2]])

    def test_wrap_identity(self, cubic_box):
        pts = np.array([[5.0, 5.0, 5.0]], dtype=np.float64)
        wrapped, shifts = cubic_box.wrap(pts)
        np.testing.assert_allclose(wrapped, pts, atol=1e-4)
        np.testing.assert_array_equal(shifts, [[0, 0, 0]])


class TestBoxDisplacement:
    def test_delta_no_mic(self, cubic_box):
        p1 = np.array([[0.1, 0.0, 0.0]], dtype=np.float64)
        p2 = np.array([[9.9, 0.0, 0.0]], dtype=np.float64)
        d = cubic_box.delta(p1, p2, minimum_image=False)
        assert pytest.approx(d[0, 0], abs=1e-3) == 9.8

    def test_delta_with_mic(self, cubic_box):
        p1 = np.array([[0.1, 0.0, 0.0]], dtype=np.float64)
        p2 = np.array([[9.9, 0.0, 0.0]], dtype=np.float64)
        d = cubic_box.delta(p1, p2, minimum_image=True)
        assert abs(d[0, 0]) < 1.0

    def test_delta_shape_mismatch(self, cubic_box):
        p1 = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        p2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            cubic_box.delta(p1, p2)

    def test_delta_1d_points_with_mic(self, cubic_box):
        p1 = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        p2 = np.array([9.9, 0.0, 0.0], dtype=np.float64)
        d = cubic_box.delta(p1, p2, minimum_image=True)
        assert d.shape == (3,)
        assert abs(d[0]) < 1.0

    def test_delta_1d_points_no_mic(self, cubic_box):
        p1 = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        p2 = np.array([9.9, 0.0, 0.0], dtype=np.float64)
        d = cubic_box.delta(p1, p2, minimum_image=False)
        assert d.shape == (3,)
        assert pytest.approx(d[0], abs=1e-3) == 9.8

    def test_delta_rejects_mixed_1d_and_nx3(self, cubic_box):
        p1 = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        p2 = np.array([[9.9, 0.0, 0.0]], dtype=np.float64)
        with pytest.raises(ValueError):
            cubic_box.delta(p1, p2)

    def test_delta_rejects_bad_1d_length(self, cubic_box):
        p1 = np.array([0.1, 0.0], dtype=np.float64)
        p2 = np.array([9.9, 0.0], dtype=np.float64)
        with pytest.raises(ValueError, match=r"\(N,\s*3\)|\(3,\)"):
            cubic_box.delta(p1, p2)


class TestBoxContainment:
    def test_isin_inside(self, cubic_box, sample_points):
        inside = cubic_box.contains(sample_points)
        assert inside.shape == (5,)
        assert all(inside)

    def test_isin_outside(self, cubic_box):
        pts = np.array([[-1.0, 0.0, 0.0], [11.0, 0.0, 0.0]], dtype=np.float64)
        inside = cubic_box.contains(pts)
        assert not inside[0]
        assert not inside[1]


class TestBoxErrors:
    def test_bad_matrix_shape(self):
        with pytest.raises((TypeError, ValueError)):
            Cell.matrix(np.ones((2, 3), dtype=np.float64))

    def test_bad_lengths_length(self):
        with pytest.raises((TypeError, ValueError)):
            Cell.lengths([1.0, 2.0])

    def test_bad_origin_length(self):
        with pytest.raises((TypeError, ValueError), match="length 3"):
            Box(
                Cell.lengths([10.0, 10.0, 10.0]),
                origin=np.array([1.0, 2.0], dtype=np.float64),
            )

    def test_bad_pbc_length(self):
        with pytest.raises((TypeError, ValueError), match="length 3"):
            Box(Cell.lengths([10.0, 10.0, 10.0]), pbc=np.array([True, False]))

    def test_an_int_pbc_flag_is_refused(self):
        with pytest.raises(TypeError):
            Box(Cell.lengths([10.0, 10.0, 10.0]), pbc=(1, 1, 1))

    def test_bad_to_frac_shape(self, cubic_box):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            cubic_box.to_frac(np.ones((3, 2), dtype=np.float64))

    def test_bad_lattice_index(self, cubic_box):
        with pytest.raises(ValueError, match="0, 1, or 2"):
            cubic_box.lattice(3)

    def test_repr(self, cubic_box):
        r = repr(cubic_box)
        assert "Box" in r
        assert "1000" in r
