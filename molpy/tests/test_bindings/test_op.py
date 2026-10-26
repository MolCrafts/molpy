"""FFI seam of ``mp.op`` (assembly-07, mirrors ``crates/molrs/src/op/``).

These prove that the numeric primitives cross: arrays arrive as float64 of the
documented shape, ``Superposition`` is a frozen record whose ``freedom`` crosses as its
lowercase name, and ``SuperpositionError`` maps to ``ValueError``. They re-derive
no numerics: superposition and the eigen-gap are proven by the unit tests in
``crates/molrs/src/op/superpose.rs``. The one geometric value asserted (a pure
translation) is hand-checkable.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest

# Position, numerical (tester contract: 1e-8).
POS_TOL = 1e-8

# Three non-collinear points: the rotation is unique.
TRIANGLE = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


class TestModule:
    def test_default_gap_tol(self):
        # `op::DEFAULT_GAP_TOL`.
        assert mp.op.DEFAULT_GAP_TOL == 1e-4


class TestSuperpose:
    def test_returns_a_fit_with_float64_arrays_of_the_documented_shape(self):
        fit = mp.op.superpose(TRIANGLE, TRIANGLE + [1.0, 2.0, 3.0])

        assert isinstance(fit, mp.op.Superposition)
        assert fit.rotation.dtype == np.float64
        assert fit.rotation.shape == (3, 3)
        assert fit.translation.dtype == np.float64
        assert fit.translation.shape == (3,)
        assert fit.center.shape == (3,)
        assert isinstance(fit.rmsd, float)
        assert isinstance(fit.rho, float)

    def test_a_pure_translation_is_recovered(self):
        # Hand-derived: target = reference + t ⇒ R = I, translation = t,
        # rmsd = 0, center = target centroid = (1/3, 1/3, 0) + t.
        t = np.array([1.0, 2.0, 3.0])
        fit = mp.op.superpose(TRIANGLE, TRIANGLE + t)

        np.testing.assert_allclose(fit.rotation, np.eye(3), rtol=0, atol=POS_TOL)
        np.testing.assert_allclose(fit.translation, t, rtol=0, atol=POS_TOL)
        np.testing.assert_allclose(
            fit.center, [1.0 / 3.0 + 1.0, 1.0 / 3.0 + 2.0, 3.0], rtol=0, atol=POS_TOL
        )
        assert fit.rmsd == pytest.approx(0.0, abs=POS_TOL)

    def test_explicit_uniform_weights_equal_the_default(self):
        target = TRIANGLE + [1.0, 2.0, 3.0]
        default = mp.op.superpose(TRIANGLE, target)
        weighted = mp.op.superpose(TRIANGLE, target, weights=np.ones(3))

        np.testing.assert_array_equal(weighted.rotation, default.rotation)
        np.testing.assert_array_equal(weighted.translation, default.translation)

    @pytest.mark.parametrize(
        ("reference", "freedom", "has_axis"),
        [
            (TRIANGLE, "unique", False),
            # Two points fix every direction but the line through them.
            (TRIANGLE[:2], "spin", True),
            # One point fixes no rotation at all.
            (TRIANGLE[:1], "free", False),
        ],
        ids=["unique", "spin", "free"],
    )
    def test_freedom_crosses_as_its_lowercase_name(self, reference, freedom, has_axis):
        fit = mp.op.superpose(reference, reference + [0.5, 0.0, 0.0])

        assert fit.freedom == freedom
        if has_axis:
            assert fit.axis.dtype == np.float64
            assert fit.axis.shape == (3,)
        else:
            assert fit.axis is None

    def test_fit_is_frozen(self):
        fit = mp.op.superpose(TRIANGLE, TRIANGLE)
        with pytest.raises(AttributeError):
            fit.rmsd = 1.0

    def test_a_length_mismatch_is_a_value_error(self):
        with pytest.raises(ValueError, match="mismatch"):
            mp.op.superpose(TRIANGLE, TRIANGLE[:2])

    def test_a_weight_count_mismatch_is_a_value_error(self):
        with pytest.raises(ValueError):
            mp.op.superpose(TRIANGLE, TRIANGLE, weights=np.ones(2))

    def test_a_non_xyz_array_is_a_value_error(self):
        with pytest.raises(ValueError):
            mp.op.superpose(TRIANGLE[:, :2], TRIANGLE[:, :2])


class TestCentroid:
    def test_weighted_centroid(self):
        # Hand-derived: (1·0 + 3·2) / 4 = 1.5.
        c = mp.op.centroid(
            np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]), np.array([1.0, 3.0])
        )
        assert c.dtype == np.float64
        np.testing.assert_allclose(c, [1.5, 0.0, 0.0], rtol=0, atol=1e-12)

    def test_zero_total_weight_has_no_centroid(self):
        assert mp.op.centroid(TRIANGLE, np.zeros(3)) is None


class TestPlaceFromInternalCoords:
    """``op::place_from_internal_coords`` crosses: lists or arrays in, a
    float64 ``(3,)`` array out, angles in radians."""

    A = [0.0, 1.0, 0.0]
    B = [0.0, 0.0, 0.0]
    C = [1.52, 0.0, 0.0]

    @staticmethod
    def _angle(b, c, d):
        u, v = np.subtract(b, c), np.subtract(d, c)
        return np.arccos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v))

    @staticmethod
    def _dihedral(a, b, c, d):
        b0, b1, b2 = np.subtract(a, b), np.subtract(c, b), np.subtract(d, c)
        b1 = b1 / np.linalg.norm(b1)
        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1
        return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))

    def test_the_point_has_the_bond_angle_and_torsion_asked_for(self):
        deg = mp.core.UnitRegistry().factor("deg", "rad")
        d = mp.op.place_from_internal_coords(
            self.A, self.B, self.C, 1.33, 116.0 * deg, -75.0 * deg
        )
        assert d.dtype == np.float64 and d.shape == (3,)
        np.testing.assert_allclose(np.linalg.norm(d - self.C), 1.33, rtol=1e-12)
        np.testing.assert_allclose(
            self._angle(self.B, self.C, d), 116.0 * deg, rtol=1e-12
        )
        np.testing.assert_allclose(
            self._dihedral(self.A, self.B, self.C, d), -75.0 * deg, rtol=1e-12
        )

    def test_trans_lies_in_the_plane_opposite_a(self):
        # Hand-derived: ∠ 90°, torsion 180° from a = +y puts d at c − y.
        d = mp.op.place_from_internal_coords(
            np.array(self.A), np.array(self.B), np.array(self.C), 1.0, np.pi / 2, np.pi
        )
        np.testing.assert_allclose(d, [1.52, -1.0, 0.0], rtol=0, atol=1e-12)

    def test_collinear_points_leave_only_the_axial_component(self):
        d = mp.op.place_from_internal_coords(
            [-1.0, 0.0, 0.0], self.B, self.C, 1.0, np.pi / 2, 0.3
        )
        np.testing.assert_allclose(d, self.C, rtol=0, atol=1e-12)

    def test_a_point_not_of_three_coordinates_is_rejected(self):
        with pytest.raises((TypeError, ValueError)):
            mp.op.place_from_internal_coords([0.0, 1.0], self.B, self.C, 1.0, 1.0, 1.0)
