import pytest
import numpy.testing as npt
import molpy as mp
import numpy as np


class TestBox:
    def test_construct(self):
        """Test correct behavior for various constructor signatures"""
        box = mp.Box(1, 2, 3)
        assert box.lx == 1 and box.ly == 2 and box.lz == 3

    def test_get_length(self):
        box = mp.Box(2, 4, 5, 0, 0, 0)

        npt.assert_allclose(box.lx, 2, rtol=1e-6)
        npt.assert_allclose(box.ly, 4, rtol=1e-6)
        npt.assert_allclose(box.lz, 5, rtol=1e-6)
        npt.assert_allclose(box.length, [2, 4, 5], rtol=1e-6)
        npt.assert_allclose(box.get_inverse(), np.diag([0.5, 0.25, 0.2]), rtol=1e-6)

    def test_tilt_factor(self):
        box = mp.Box(2, 2, 2, 0, 1, 2)

        npt.assert_allclose(box.xy, 0, rtol=1e-6)
        npt.assert_allclose(box.xz, 1, rtol=1e-6)
        npt.assert_allclose(box.yz, 2, rtol=1e-6)

    def test_box_volume(self):
        box3d = mp.Box(2, 2, 2, 2, 0, 0)
        npt.assert_allclose(box3d.get_volume(), 8, rtol=1e-6)

    def test_wrap(self):

        cell = np.array([[0, 0, 4]])
        box = mp.Box(1, 1, 6)
        npt.assert_allclose(box.wrap(cell), cell, rtol=1e-6)

    def test_wrap_single_particle(self):
        box = mp.Box(2, 2, 2, 0, 0, 0)

        points = [0.1, 0, 0]
        npt.assert_allclose(box.wrap(points)[0, 0], 0.1, rtol=1e-6)

        points = np.array([-0.1, 0, 0])
        npt.assert_allclose(box.wrap(points)[0, 0], 1.9, rtol=1e-6)

    def test_wrap_multiple_particles(self):
        box = mp.Box(2, 2, 2, 0, 0, 0)

        points = [[0.1, -1, -1], [0.1, 0.5, 0]]
        expected = [[0.1, 1, 1], [0.1, 0.5, 0]]
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

    def test_wrap_triclinic_multiple_particles(self):
        box = mp.Box(2, 2, 2, 2, 0, 0)

        points = [[0, -1, -1], [0, 0.5, 0]]
        expected = [[2, 1, 1], [2, 0.5, 0]]
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

    def test_wrap_multiple_images(self):
        box = mp.Box(2, 2, 2, 2, 0, 0)

        points = [[10, -5, -5], [0, 0.5, 0]]
        expected = [[2, 1, 1], [2, 0.5, 0]]
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

        points = np.array(points)
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

    def test_wrap(self):
        box = mp.Box(2, 2, 2, 2, 0, 0)
        points = [[10, -5, -5], [0, 0.5, 0]]
        expected = [[2, 1, 1], [2, 0.5, 0]]
        npt.assert_allclose(box.wrap(points), expected, rtol=1e-6)

    def test_unwrap(self):
        box = mp.Box(2, 2, 2, 2, 0, 0)

        points = [0, -1, -1]
        imgs = [1, 0, 0]
        npt.assert_allclose(box.unwrap(points, imgs), [[2, -1, -1]], rtol=1e-6)

        points = [[0, -1, -1], [0, 0.5, 0]]
        imgs = [[1, 0, 0], [1, 1, 0]]
        expected = [[2, -1, -1], [4, 2.5, 0]]
        npt.assert_allclose(box.unwrap(points, imgs), expected, rtol=1e-6)

        points = np.array(points)
        imgs = np.array(imgs)
        npt.assert_allclose(box.unwrap(points, imgs), expected, rtol=1e-6)

        # Test broadcasting one image with multiple vectors
        box = mp.Box.cube(1)

        points = [[10, 0, 0], [11, 0, 0]]
        imgs = [10, 1, 2]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6
        )

        # Test broadcasting one vector with multiple images
        box = mp.Box.cube(1)

        points = [10, 0, 0]
        imgs = [[10, 1, 2], [11, 1, 2]]
        npt.assert_allclose(
            box.unwrap(points, imgs), [[20, 1, 2], [21, 1, 2]], rtol=1e-6
        )

    def test_images_3d(self):
        box = mp.Box(2, 2, 2, 0, 0, 0)
        points = np.array([[50, 40, 30], [-10, 0, 0]])
        images = box.get_image(points)
        npt.assert_equal(images, np.array([[25, 20, 15], [-5, 0, 0]]))
