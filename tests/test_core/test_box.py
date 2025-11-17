import numpy as np
import numpy.testing as npt

from molpy import Box


class TestBoxConstruction:
    def test_matrix_construction(self):
        matrix = np.diag([1, 2, 3])
        box = Box(matrix)
        assert box.style == Box.Style.ORTHOGONAL
        npt.assert_allclose(box.lx, 1)
        npt.assert_allclose(box.ly, 2)
        npt.assert_allclose(box.lz, 3)

    def test_cubic_and_orth(self):
        b1 = Box.cubic(5.0)
        assert b1.style == Box.Style.ORTHOGONAL
        assert b1.lx == b1.ly == b1.lz == 5.0
        b2 = Box.orth([2, 3, 4])
        npt.assert_allclose(b2.l, [2, 3, 4])

    def test_triclinic_basic(self):
        lengths = [2, 3, 4]
        tilts = [0.5, 1.0, 1.5]
        box = Box.tric(lengths, tilts)
        assert box.style == Box.Style.TRICLINIC


class TestBoxProperties:
    def test_lengths_and_tilts(self):
        box = Box.tric([2, 4, 5], [1, 0, 0])
        npt.assert_allclose(box.lx, 2)
        npt.assert_allclose(box.ly, 4)
        npt.assert_allclose(box.lz, 5)
        npt.assert_allclose(box.l_inv, [0.5, 0.25, 0.2])
        box.xy = 2
        npt.assert_allclose(box.xy, 2)

    def test_bounds_and_volume(self):
        box = Box.orth([2, 3, 4])
        bounds = box.bounds
        expected = np.array([[0, 0, 0], [2, 3, 4]])
        npt.assert_allclose(bounds, expected)
        assert abs(float(box.volume) - 24.0) < 1e-10

    def test_periodic_flags(self):
        box = Box.orth([1, 2, 3])
        assert box.periodic
        box.periodic_x = False
        assert not box.periodic
        box.periodic = True
        assert box.periodic


class TestBoxOps:
    def test_mul_and_repr(self):
        box = Box.tric([2, 3, 4], [1, 0.5, 0.1])
        box2 = box * 2
        npt.assert_allclose(box2.lx, 4)
        npt.assert_allclose(box2.ly, 6)
        npt.assert_allclose(box2.lz, 8)
        assert "Box" in repr(box)
