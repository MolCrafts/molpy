"""``mp.core.Trace`` FFI seam: construction, dtype and shape at the boundary."""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest


def test_trace_keeps_the_points_as_float64_in_order() -> None:
    points = np.array([[0.0, 1.0, 2.0], [3.5, -4.0, 5.25]], dtype=np.float64)

    trace = mp.core.Trace(points)

    assert len(trace) == 2
    out = trace.points
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, points)


def test_an_empty_trace_has_no_points() -> None:
    trace = mp.core.Trace(np.zeros((0, 3), dtype=np.float64))

    assert len(trace) == 0
    assert trace.points.shape == (0, 3)


def test_a_point_that_is_not_3d_is_a_value_error() -> None:
    with pytest.raises(ValueError, match=r"\(k, 3\)"):
        mp.core.Trace(np.zeros((2, 2), dtype=np.float64))


def test_trace_is_frozen_and_built_only_through_init() -> None:
    trace = mp.core.Trace(np.zeros((1, 3), dtype=np.float64))

    with pytest.raises(AttributeError):
        trace.points = np.ones((1, 3))  # type: ignore[misc]
