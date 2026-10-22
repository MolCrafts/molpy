import molpy as mp
import numpy as np
import pytest
from molpy.core import Box, Cell


def _cube(edge: float, **kwargs) -> Box:
    return Box(Cell.lengths([edge, edge, edge]), **kwargs)


def test_box_exposes_native_minimum_image_geometry():
    box = _cube(10.0)
    r1 = np.array([1.0, 1.0, 1.0])
    r2 = np.array([9.0, 1.0, 1.0])
    np.testing.assert_allclose(box.delta(r1, r2, minimum_image=True), [-2.0, 0.0, 0.0])
    np.testing.assert_allclose(box.distances(r1[None], r2[None]), [2.0])


def test_box_exposes_native_face_distances_and_corners():
    box = Box(Cell.lengths(np.array([10.0, 20.0, 30.0])))
    np.testing.assert_allclose(box.nearest_plane_distance, [10.0, 20.0, 30.0])
    corners = box.corners()
    assert corners.shape == (8, 3)
    np.testing.assert_allclose(corners.min(axis=0), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(corners.max(axis=0), [10.0, 20.0, 30.0])


def test_wrap_shifts_and_unwrap_round_trip_natively():
    box = _cube(10.0)
    unwrapped = np.array([[21.0, -9.0, 5.0], [2.0, 3.0, 34.0]])
    wrapped, images = box.wrap(unwrapped)
    np.testing.assert_array_equal(images, [[2, -1, 0], [0, 0, 3]])
    np.testing.assert_allclose(wrapped, [[1.0, 1.0, 5.0], [2.0, 3.0, 4.0]])
    np.testing.assert_allclose(box.unwrap(wrapped, images), unwrapped)


def test_wrap_shifts_are_int32_like_the_frame_image_columns():
    box = _cube(10.0)
    _, images = box.wrap(np.array([[21.0, -9.0, 5.0]]))
    assert images.dtype == np.int32
    np.testing.assert_array_equal(images, [[2, -1, 0]])


def test_wrap_leaves_a_non_periodic_axis_alone():
    box = _cube(10.0, pbc=(True, False, True))
    wrapped, images = box.wrap(np.array([[21.0, -9.0, 5.0]]))
    np.testing.assert_allclose(wrapped, [[1.0, -9.0, 5.0]])
    np.testing.assert_array_equal(images, [[2, 0, 0]])


def test_unwrap_accepts_the_int32_image_columns_of_a_frame():
    # ix/iy/iz are the schema's integer type (int32), as a LAMMPS reader
    # stores them; unwrap must take them without a cast.
    atoms = mp.core.Block(
        {
            "x": np.array([1.0, 9.0]),
            "y": np.array([2.0, 5.0]),
            "z": np.array([3.0, 0.5]),
            "ix": np.array([1, 0], dtype=np.int32),
            "iy": np.array([0, -1], dtype=np.int32),
            "iz": np.array([0, 2], dtype=np.int32),
        }
    )
    frame = mp.core.Frame({"atoms": atoms}, box=_cube(10.0))
    atoms = frame["atoms"]

    unwrapped = frame.box.unwrap(atoms["x", "y", "z"], atoms["ix", "iy", "iz"])

    # xyz + L * image, L = 10 on every axis.
    np.testing.assert_allclose(
        unwrapped, [[11.0, 2.0, 3.0], [9.0, -5.0, 20.5]], atol=1e-12
    )


def test_bounding_and_batched_geometry():
    points = np.array([[0.0, -1.0, 0.0], [2.0, 3.0, 4.0]])
    box = Box.bounding(
        points,
        np.array([1.0, 2.0, 3.0]),
        np.array([True, True, True]),
    )
    np.testing.assert_allclose(box.origin, [-1.0, -3.0, -3.0])
    np.testing.assert_allclose(box.lengths, [4.0, 8.0, 10.0])
    assert list(box.pbc) == [True, True, True]

    left = np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]])
    right = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        box.pairwise_delta(left, right), [[[1, 0, 0]], [[1.5, 0, 0]]]
    )
    np.testing.assert_allclose(box.pairwise_distances(left, right), [[1.0], [1.5]])


def test_bounding_takes_a_frames_coordinates_and_a_free_boundary():
    frame = mp.core.Frame()
    atoms = mp.core.Block()
    atoms.insert("x", np.array([0.0, 2.0]))
    atoms.insert("y", np.array([-1.0, 3.0]))
    atoms.insert("z", np.array([0.0, 4.0]))
    frame["atoms"] = atoms
    points = np.array([[0.0, -1.0, 0.0], [2.0, 3.0, 4.0]])

    from_frame = Box.bounding(frame.coords, (1.0, 1.0, 1.0), (False, False, False))
    from_points = Box.bounding(points, np.array([1.0, 1.0, 1.0]), (False,) * 3)
    np.testing.assert_allclose(from_frame.origin, [-1.0, -2.0, -1.0])
    np.testing.assert_allclose(from_frame.lengths, [4.0, 6.0, 6.0])
    assert from_frame.is_free
    assert from_frame.approx_eq(from_points, 0.0)


def test_bounding_refuses_no_points_a_bad_padding_and_a_zero_extent():
    with pytest.raises(ValueError, match="at least one point"):
        Box.bounding(np.zeros((0, 3)), (1.0, 1.0, 1.0), (False,) * 3)
    with pytest.raises((TypeError, ValueError), match="length 3"):
        Box.bounding(np.zeros((1, 3)), np.array([1.0, 1.0]), (False,) * 3)
    with pytest.raises(ValueError, match=">= 0"):
        Box.bounding(np.zeros((1, 3)), (-1.0, 1.0, 1.0), (False,) * 3)
    with pytest.raises(ValueError, match="zero extent along axis 1"):
        Box.bounding(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]), (0.0, 0.0, 0.0), (False,) * 3
        )
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        Box.bounding(np.zeros((2, 2)), (1.0, 1.0, 1.0), (False,) * 3)


def test_approx_eq_uses_an_absolute_tolerance_and_exact_pbc():
    a = _cube(10.0)
    b = _cube(10.0 + 1e-6)
    assert a.approx_eq(b, 1e-5)
    assert not a.approx_eq(b, 1e-7)
    assert not a.approx_eq(_cube(10.0, pbc=np.array([True, True, False])), 1.0)
    with pytest.raises(ValueError):
        a.approx_eq(b, -1.0)


def test_transformed_preserves_origin_and_pbc():
    box = Box(
        Cell.lengths(np.array([2.0, 3.0, 4.0])),
        origin=np.array([1.0, 2.0, 3.0]),
        pbc=np.array([True, False, True]),
    )
    transformed = box.transformed(np.diag([2.0, 1.0, 0.5]))
    np.testing.assert_allclose(transformed.matrix, np.diag([4.0, 3.0, 2.0]))
    np.testing.assert_allclose(transformed.origin, box.origin)
    np.testing.assert_array_equal(transformed.pbc, box.pbc)
