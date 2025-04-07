# Copyright (c) 2010-2025 The Regents of the University of Michigan
# This file is from the freud project, released under the BSD 3-Clause License.

from collections import namedtuple

import numpy as np
import numpy.testing as npt
import pytest

import molpy as mp


def from_freud_to_molpy(Lx, Ly, Lz, xy, xz, yz):
    return mp.Box(np.array([[Lx, xy, xz], [0, Ly, yz], [0, 0, Lz]]))


class TestBox:

    def test_init_volume(self):
        assert mp.Box().style == mp.Box.Style.FREE
        assert mp.Box(np.diag([1, 2, 3])).style == mp.Box.Style.ORTHOGONAL
        assert mp.Box.cubic(1).style == mp.Box.Style.ORTHOGONAL
        assert mp.Box.orth([1, 2, 3]).style == mp.Box.Style.ORTHOGONAL
        assert mp.Box.tric([1, 2, 3], [0.5, 1, 1.5]).style == mp.Box.Style.TRICLINIC

    def test_get_length(self):
        box = mp.Box.tric([2, 4, 5], [1, 0, 0])

        npt.assert_allclose(box.lx, 2, rtol=1e-6)
        npt.assert_allclose(box.ly, 4, rtol=1e-6)
        npt.assert_allclose(box.lz, 5, rtol=1e-6)
        npt.assert_allclose(box.l, np.array([2, 4, 5]), rtol=1e-6)
        npt.assert_allclose(box.l_inv, np.array([0.5, 0.25, 0.2]), rtol=1e-6)

    def test_set_length(self):
        # Make sure we can change the lengths of the box after its creation
        box = mp.Box.tric([1, 2, 3], [1, 0, 0])

        box.lx = 4
        box.ly = 5
        box.lz = 6

        npt.assert_allclose(box.lx, 4, rtol=1e-6)
        npt.assert_allclose(box.ly, 5, rtol=1e-6)
        npt.assert_allclose(box.lz, 6, rtol=1e-6)

        box.lengths = [7, 8, 9]
        npt.assert_allclose(box.lengths, np.array([7, 8, 9]), rtol=1e-6)

        with pytest.raises(AssertionError):
            box.lengths = [1, 2, 3, 4]

        with pytest.raises(AssertionError):
            box.lengths = [1, 2]

    def test_get_tilt_factor(self):
        box = mp.Box.tric([2, 2, 2], [1, 2, 3])

        npt.assert_allclose(box.xy, 1, rtol=1e-6)
        npt.assert_allclose(box.xz, 2, rtol=1e-6)
        npt.assert_allclose(box.yz, 3, rtol=1e-6)

    def test_set_tilt_factor(self):
        box = mp.Box.tric([2, 2, 2], [1, 2, 3])
        box.xy = 4
        box.xz = 5
        box.yz = 6

        npt.assert_allclose(box.xy, 4, rtol=1e-6)
        npt.assert_allclose(box.xz, 5, rtol=1e-6)
        npt.assert_allclose(box.yz, 6, rtol=1e-6)

    def test_wrap_single_particle(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])

        points = [0, -1, -1]
        npt.assert_allclose(box.wrap(points)[0], 1, rtol=1e-6)

        with pytest.raises(ValueError):
            box.wrap([1, 2])

    def test_wrap_multiple_particles(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])

        points = [[0, -1, -1], [0, 0.5, 0]]
        wrapped = box.wrap(points)
        npt.assert_allclose(wrapped[0, 0], 1, rtol=1e-6)
        npt.assert_allclose(wrapped[1, 0], 2, rtol=1e-6)

    def test_wrap(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])
        points = [[10, -5, -5], [0, 0.5, 0]]
        npt.assert_allclose(box.wrap(points), [[1, 1, 1], [2, 0.5, 0]], rtol=1e-6)

    def test_wrap_with_non_zero_original(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0], origin=[-1, -1, -1])
        points = [[10, -5, -5], [0, 0.5, 0]]
        points = np.array(points, dtype=np.float32)
        wrapped = box.wrap(points)
        npt.assert_allclose(wrapped, [[0, -1, -1], [0, 0.5, 0]], rtol=1e-6)

    def test_unwrap(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0])

        points = [0, -1, -1]
        imgs = [1, 0, 0]
        npt.assert_allclose(box.unwrap(points, imgs), [2, -1, -1], rtol=1e-6)

        points = [[0, -1, -1], [0, 0.5, 0]]
        imgs = [[1, 0, 0], [1, 1, 0]]
        npt.assert_allclose(box.unwrap(points, imgs)[0, 0], 2, rtol=1e-6)

        points = np.array(points)
        imgs = np.array(imgs)
        npt.assert_allclose(box.unwrap(points, imgs)[0, 0], 2, rtol=1e-6)

        with pytest.raises(ValueError):
            box.unwrap(points, imgs[..., np.newaxis])

        with pytest.raises(ValueError):
            box.unwrap(points[:, :2], imgs)

        # Test broadcasting one image with multiple vectors
        box = mp.Box.cubic(1)

        points = [[10, 0, 0], [11, 0, 0]]
        imgs = [10, 1, 2]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6
        )

        # Test broadcasting one vector with multiple images
        box = mp.Box.cubic(1)

        points = [10, 0, 0]
        imgs = [[10, 1, 2], [11, 1, 2]]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6
        )

    def test_unwrap_with_non_zero_original(self):
        box = mp.Box.tric([2, 2, 2], [1, 0, 0], origin=[-1, -1, -1])
        points = [[0, -1, -1], [0, 0.5, 0]]
        imgs = [[1, 0, 0], [1, 1, 0]]
        points = np.array(points)
        imgs = np.array(imgs)
        npt.assert_allclose(
            box.unwrap(points, imgs), [[2, -1, -1], [3, 2.5, 0]], rtol=1e-6
        )

    def test_images_3d(self):
        box = mp.Box.tric([2, 2, 2], [0, 0, 0])
        points = np.array([[50, 40, 30], [-10, 0, 0]])
        images = np.array([box.get_images(vec) for vec in points])
        npt.assert_equal(images, np.array([[25, 20, 15], [-5, 0, 0]]))
        images = box.get_images(points)
        npt.assert_equal(images, np.array([[25, 20, 15], [-5, 0, 0]]))

    # def test_center_of_mass(self):
    #     box = mp.Box.cubic(5)

    #     npt.assert_allclose(box.center_of_mass([[0, 0, 0]]), [0, 0, 0], atol=1e-6)
    #     npt.assert_allclose(box.center_of_mass([[1, 1, 1]]), [1, 1, 1], atol=1e-6)
    #     npt.assert_allclose(
    #         box.center_of_mass([[1, 1, 1], [2, 2, 2]]), [1.5, 1.5, 1.5], atol=1e-6
    #     )
    #     npt.assert_allclose(
    #         box.center_of_mass([[-2, -2, -2], [2, 2, 2]]), [-2.5, -2.5, -2.5], atol=1e-6
    #     )
    #     npt.assert_allclose(
    #         box.center_of_mass([[-2.2, -2.2, -2.2], [2, 2, 2]]),
    #         [2.4, 2.4, 2.4],
    #         atol=1e-6,
    #     )

    # def test_center_of_mass_weighted(self):
    #     box = mp.Box.tric(5)

    #     points = [[0, 0, 0], -box.L / 4]
    #     masses = [2, 1]
    #     phases = np.exp(2 * np.pi * 1j * box.make_fractional(points))
    #     com_angle = np.angle(phases.T @ masses / np.sum(masses))
    #     com = box.make_absolute(com_angle / (2 * np.pi))
    #     npt.assert_allclose(box.center_of_mass(points, masses), com, atol=1e-6)

    # def test_center(self):
    #     box = mp.Box.cubic(5)

    #     npt.assert_allclose(box.center([[0, 0, 0]]), [[0, 0, 0]], atol=1e-6)
    #     npt.assert_allclose(box.center([[1, 1, 1]]), [[0, 0, 0]], atol=1e-6)
    #     npt.assert_allclose(
    #         box.center([[1, 1, 1], [2, 2, 2]]),
    #         [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    #         atol=1e-6,
    #     )
    #     npt.assert_allclose(
    #         box.center([[-2, -2, -2], [2, 2, 2]]),
    #         [[0.5, 0.5, 0.5], [-0.5, -0.5, -0.5]],
    #         atol=1e-6,
    #     )

    # def test_center_weighted(self):
    #     box = mp.Box.cubic(5)

    #     points = [[0, 0, 0], -box.L / 4]
    #     masses = [2, 1]
    #     phases = np.exp(2 * np.pi * 1j * box.make_fractional(points))
    #     com_angle = np.angle(phases.T @ masses / np.sum(masses))
    #     com = box.make_absolute(com_angle / (2 * np.pi))
    #     npt.assert_allclose(
    #         box.center(points, masses), box.wrap(points - com), atol=1e-6
    #     )

    #     # Make sure the center of mass is not (0, 0, 0) if ignoring masses
    #     assert not np.allclose(box.center_of_mass(points), [0, 0, 0], atol=1e-6)

    def test_absolute_coordinates(self):
        box = mp.Box.orth([2, 2, 2], central=True)
        f_point = np.array([[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]])
        point = np.array([[0, -0.5, 0.5], [-1, -1, -1], [0, 0, 0]])

        testcoordinates = box.make_absolute(f_point)
        npt.assert_equal(testcoordinates, point)
        testcoordinates = box.make_absolute(f_point)
        npt.assert_equal(testcoordinates, point)

    def test_fractional_coordinates(self):
        box = mp.Box.orth([2, 2, 2], central=True)
        f_point = np.array([[0.5, 0.25, 0.75], [0, 0, 0], [0.5, 0.5, 0.5]])
        point = np.array([[0, -0.5, 0.5], [-1, -1, -1], [0, 0, 0]])

        testfraction = np.array([box.make_fractional(vec) for vec in point])
        npt.assert_equal(testfraction, f_point)

        testfraction = box.make_fractional(point)

        npt.assert_equal(testfraction, f_point)

    def test_vectors(self):
        """Test getting lattice vectors"""
        b_list = [1, 2, 3, 0.1, 0.2, 0.3]
        Lx, Ly, Lz = b_list[:3]
        xy, xz, yz = b_list[3:]
        box = mp.Box.tric((Lx, Ly, Lz), (xy, xz, yz))
        npt.assert_allclose(box.a, [Lx, 0, 0])
        npt.assert_allclose(box.b, [xy, Ly, 0])
        npt.assert_allclose(box.c, [xz, yz, Lz])

    def test_periodic(self):
        box = mp.Box.orth([1, 2, 3])
        npt.assert_array_equal(box.periodic, True)
        assert box.periodic_x
        assert box.periodic_y
        assert box.periodic_z

        # Test setting all flags together
        box.periodic = False
        npt.assert_array_equal(box.periodic, False)
        assert not box.periodic_x
        assert not box.periodic_y
        assert not box.periodic_z

        # Test setting flags as a list
        box.periodic = [True, True, True]
        npt.assert_array_equal(box.periodic, True)

        # Test setting each flag separately
        box.periodic_x = False
        box.periodic_y = False
        box.periodic_z = False
        assert not box.periodic_x
        assert not box.periodic_y
        assert not box.periodic_z

        box.periodic = True
        npt.assert_array_equal(box.periodic, True)

    def test_equal(self):
        box1 = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box1_copy = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        assert box1 == box1_copy
        box2 = mp.Box.tric([2, 2, 2], [1, 0, 0])
        assert box1 != box2
        box1_nonperiodic = mp.Box.tric(
            [
                2,
                2,
                2,
            ],
            [1, 0.5, 0.1],
        )
        box1_nonperiodic.periodic = [False, False, False]
        assert box1 != box1_nonperiodic

    def test_repr(self):
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        assert repr(box).startswith("<Triclinic Box:")

    def test_str(self):
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box2 = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        assert str(box) == str(box2)

    def test_to_dict(self):
        """Test converting box to dict"""
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box2 = box.to_dict()
        box_dict_values = {
            "xlo",
            "xhi",
            "ylo",
            "yhi",
            "zlo",
            "zhi",
            "xy",
            "xz",
            "yz",
            "x_pbc",
            "y_pbc",
            "z_pbc",
        }
        assert set(box2.keys()) == box_dict_values

    def test_from_box(self):
        """Test various methods of initializing a box"""
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box2 = mp.Box.from_box(box)
        assert box == box2

    def test_standard_orthogonal_box(self):
        box = mp.Box.tric((1, 2, 3), (0, 0, 0))
        lengths, angles = box.to_lengths_angles()
        npt.assert_allclose(lengths, (1, 2, 3))
        npt.assert_allclose(angles, (90, 90, 90))

    def test_to_and_from_box_lengths_and_angles(self):

        abc = np.array(
            [
                np.random.uniform(0, 100000),
                np.random.uniform(0, 100000),
                np.random.uniform(0, 100000),
            ]
        )
        angles = np.array(
            [
                np.random.uniform(0, 180),
                np.random.uniform(0, 180),
                np.random.uniform(0, 180),
            ]
        )
        alpha, beta, gamma = np.deg2rad(angles)
        cos_alpha, cos_beta, cos_gamma = np.cos([alpha, beta, gamma])
        cos_check = (
            cos_alpha**2
            + cos_beta**2
            + cos_gamma**2
            - 2 * cos_alpha * cos_beta * cos_gamma
        )

        if cos_check >= 1.0:
            with pytest.raises(ValueError):
                mp.Box.from_lengths_angles(abc, angles)
        else:
            box = mp.Box.from_lengths_angles(abc, angles)
            box_lengths, box_angles = box.to_lengths_angles()

            np.testing.assert_allclose(
                box_lengths,
                abc,
                rtol=1e-5,
                atol=1e-14,
            )
            np.testing.assert_allclose(
                box_angles,
                angles,
                rtol=1e-5,
                atol=1e-14,
            )

    def test_matrix(self):
        box = mp.Box.tric([2, 2, 2], [1, 0.5, 0.1])
        box2 = mp.Box(box.matrix)
        assert np.isclose(box.matrix, box2.matrix).all()

    def test_cubic(self):
        L = 10.0
        cubic = mp.Box.cubic(length=L)
        assert cubic.lx == L
        assert cubic.ly == L
        assert cubic.lz == L
        assert cubic.xy == 0
        assert cubic.xz == 0
        assert cubic.yz == 0

    def test_multiply(self):
        box = mp.Box.tric([2, 3, 4], [1, 0.5, 0.1])
        box2 = box * 2
        assert np.isclose(box2.lx, 4)
        assert np.isclose(box2.ly, 6)
        assert np.isclose(box2.lz, 8)
        assert np.isclose(box2.xy, 2)
        assert np.isclose(box2.xz, 1)
        assert np.isclose(box2.yz, 0.2)
        box3 = 2 * box
        assert box2 == box3

    def test_plot_3d(self):
        box = mp.Box.tric([2, 3, 4], [1, 0.5, 0.1])
        box.plot()

    def test_compute_distances_3d(self):
        box = mp.Box.tric([2, 3, 4], [1, 0, 0])
        points = np.array([[0, 0, 0], [-2.2, -1.3, 2]])
        query_points = np.array(
            [[-0.5, -1.3, 2.0], [0.5, 0, 0], [-2.2, -1.3, 2.0], [0, 0, 0.2]]
        )
        point_indices = np.array([1, 0, 1, 0])  # 0, 1, 0
        query_point_indices = np.array([0, 1, 2, 3])  #   1, 2, 3
        distances = box.dist(query_points[query_point_indices], points[point_indices])
        npt.assert_allclose(distances, [0.3, 0.5, 0.0, 0.2], rtol=1e-6)

    def test_compute_all_distances_3d(self):
        box = mp.Box.tric([2, 3, 4], [1, 0, 0])
        points = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        query_points = np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        distances = box.dist_all(points, query_points)
        assert distances.size == len(query_points) * len(points)
        npt.assert_allclose(
            distances, [[1.0, 0.0, 1.0], [np.sqrt(2), 1.0, 0.0]], rtol=1e-6
        )

    def test_isin(self):

        box = mp.Box.tric(lengths=[2.0, 2.0, 2.0], tilts=[1.0, 0.0, 0.0])

        points = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0, 0], [1.0, 1.0, 0.0], [0.5, 1.75, 0.0]]
        )
        npt.assert_allclose(box.isin(points), [True, False, True, False], rtol=1e-6)
