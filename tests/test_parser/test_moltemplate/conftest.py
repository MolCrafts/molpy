"""Shared moltemplate fixtures: array-replication documents and their goldens."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

Coordinates = list[tuple[float, float, float]]


def _rot_copy(k: int) -> tuple[float, float, float]:
    # (2, 0, 0) turned k * 30 degrees about the z axis through the centre (1, 0, 0).
    angle = math.radians(30.0 * k)
    return (1.0 + math.cos(angle), math.sin(angle), 0.0)


@pytest.fixture(scope="session")
def array_transforms_dir(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "moltemplate" / "array_transforms"


@pytest.fixture(scope="session")
def array_transform_goldens() -> dict[str, Coordinates]:
    """Copy k of ``[N].op(args)`` carries ``op`` applied k times; copy 0 is the template.

    Hand-derived from each fixture's header comment (one atom per copy, copies
    in replication order).
    """
    return {
        "array_move.lt": [(1.0, 0.0, 0.0), (3.0, 0.5, 0.0), (5.0, 1.0, 0.0)],
        "array_rot.lt": [_rot_copy(k) for k in range(4)],
        "array_scale.lt": [(1.0, 2.0, 3.0), (3.0, 6.0, 9.0), (9.0, 18.0, 27.0)],
    }
