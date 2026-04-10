"""Tests for the Compute base class."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from molpy.tool.base import Compute


# ---------------------------------------------------------------------------
# Fixtures: concrete Compute subclass for testing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DummyCompute(Compute):
    """A dummy compute for testing."""

    threshold: float
    max_iter: int = 100

    def run(self, data: NDArray, scale: float = 1.0) -> NDArray:
        return data * self.threshold * scale


# ---------------------------------------------------------------------------
# Compute interface tests
# ---------------------------------------------------------------------------


class TestCompute:
    def test_call_delegates_to_run(self):
        c = DummyCompute(threshold=2.0)
        data = np.array([1.0, 2.0, 3.0])
        result = c(data)
        np.testing.assert_array_equal(result, data * 2.0)

    def test_run_directly(self):
        c = DummyCompute(threshold=3.0)
        data = np.array([1.0, 2.0])
        result = c.run(data, scale=2.0)
        np.testing.assert_array_equal(result, data * 3.0 * 2.0)

    def test_frozen_immutability(self):
        c = DummyCompute(threshold=1.0)
        with pytest.raises(AttributeError):
            c.threshold = 2.0  # type: ignore[misc]

    def test_get_node_id(self):
        assert DummyCompute.get_node_id() == "DummyCompute"

    def test_instance_reuse(self):
        """Same instance can be called with different data."""
        c = DummyCompute(threshold=2.0)
        r1 = c(np.array([1.0]))
        r2 = c(np.array([5.0]))
        np.testing.assert_array_equal(r1, np.array([2.0]))
        np.testing.assert_array_equal(r2, np.array([10.0]))
