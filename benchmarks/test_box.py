"""molpy.core.Box benchmarks: construction and per-point transforms."""

from __future__ import annotations

import numpy as np
import pytest

import molpy as mp

from conftest import BOX_LEN

pytestmark = pytest.mark.benchmark


def test_box_cubic_construct(benchmark) -> None:
    box = benchmark(mp.Box.cubic, BOX_LEN)
    assert box.volume == pytest.approx(BOX_LEN**3)


def test_box_make_fractional(benchmark, points: np.ndarray) -> None:
    box = mp.Box.cubic(BOX_LEN)
    out = benchmark(box.make_fractional, points)
    assert out.shape == points.shape


def test_box_make_absolute(benchmark, points: np.ndarray) -> None:
    box = mp.Box.cubic(BOX_LEN)
    frac = box.make_fractional(points)
    out = benchmark(box.make_absolute, frac)
    assert out.shape == points.shape


def test_box_wrap(benchmark, points: np.ndarray) -> None:
    box = mp.Box.cubic(BOX_LEN)
    out = benchmark(box.wrap, points)
    assert out.shape == points.shape
