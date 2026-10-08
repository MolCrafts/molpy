"""Space-group expansion for the crystal builder (``mp.builder.SpaceGroup``)."""

import numpy as np
import pytest

from molpy.builder import SpaceGroup
from molpy.builder._symmetry import parse_triplet


class TestSpaceGroup:
    def test_identity_generator_has_order_one(self):
        group = SpaceGroup.from_generators(["x,y,z"])
        assert group.order == 1


def test_parse_triplet_identity():
    R, t = parse_triplet("x,y,z")
    assert np.allclose(R, np.eye(3))
    assert np.allclose(t, 0)


def test_parse_triplet_fraction_and_sign():
    R, t = parse_triplet("-y+1/2, x+1/2, z+1/2")
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.allclose(t, [0.5, 0.5, 0.5])


def test_translation_wrapped_into_unit_cell():
    _, t = parse_triplet("x+5/4, y, z-3/4")
    assert np.allclose(t, [0.25, 0.0, 0.25])  # 5/4 -> 1/4, -3/4 -> 1/4
    assert all(0 <= ti < 1 for ti in t)


def test_bad_triplet_rejected():
    with pytest.raises(ValueError):
        parse_triplet("x,y")
