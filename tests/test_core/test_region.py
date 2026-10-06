"""Regions are molrs's (``mp.Cuboid is molrs.spatial.Cuboid``).

molpy keeps no region classes: masking a block, filtering it, composing
regions with ``&`` / ``|`` / ``~`` and selecting by distance are native. These
tests pin the molpy-facing contract, including the replacements for the
deleted ``BoxRegion`` / ``SphereRegion`` / ``Cube`` and ``DistanceSelector``.
"""

import molrs
import numpy as np
import pytest

import molpy as mp


def _block() -> mp.Block:
    return mp.Block(
        {
            "x": np.array([0.5, 3.0, 1.0, 0.0]),
            "y": np.array([0.5, 3.0, 1.0, 0.0]),
            "z": np.array([0.5, 3.0, 1.0, 0.0]),
            "type_id": np.array([1, 1, 2, 2]),
        }
    )


@pytest.mark.parametrize(
    "name",
    ["Cuboid", "Sphere", "HalfSpace", "Parallelepiped", "Cylinder", "Region"],
)
def test_region_names_are_molrs_objects(name):
    assert getattr(mp, name) is getattr(molrs.spatial, name)


def test_cuboid_mask_and_filter():
    box = mp.Cuboid([0.0, 0.0, 0.0], [2.0, 2.0, 2.0])
    assert box.mask(_block()).tolist() == [True, False, True, True]
    assert box(_block()).nrows == 3


def test_cube_and_its_geometry():
    cube = mp.Cuboid.cube(2.0, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cube.origin, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(cube.lengths, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(cube.bounds(), [[1.0, 3.0]] * 3)


def test_sphere_geometry_and_mask():
    sphere = mp.Sphere([0.0, 0.0, 0.0], 1.0)
    np.testing.assert_allclose(sphere.center, [0.0, 0.0, 0.0])
    assert sphere.radius == 1.0
    assert sphere.mask(_block()).tolist() == [True, False, False, True]


def test_composition():
    box = mp.Cuboid.cube(2.0)
    sphere = mp.Sphere([0.0, 0.0, 0.0], 1.0)
    assert (box & sphere).mask(_block()).tolist() == [True, False, False, True]
    assert (box | sphere).mask(_block()).tolist() == [True, False, True, True]
    assert (box & ~sphere).mask(_block()).tolist() == [False, False, True, False]
    assert isinstance(box & sphere, mp.Region)


def test_a_spherical_shell_replaces_the_distance_selector():
    block = mp.Block(
        {"x": np.array([0.0, 0.5, 1.0, 2.0]), "y": np.zeros(4), "z": np.zeros(4)}
    )
    shell = mp.Sphere([0.0, 0.0, 0.0], 1.0) & ~mp.Sphere([0.0, 0.0, 0.0], 0.5)
    inside = shell.mask(block).tolist()
    assert inside[0] is False and inside[3] is False
    assert inside[2] is True


def test_a_slab_replaces_the_coordinate_range_selector():
    slab = mp.HalfSpace([-1.0, 0.0, 0.0], [0.4, 0.0, 0.0]) & mp.HalfSpace(
        [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]
    )
    assert slab.mask(_block()).tolist() == [True, False, True, False]


def test_a_selector_composes_with_a_region():
    picked = mp.AtomTypeSelector(1, field="type_id") & mp.Cuboid.cube(2.0)
    assert picked.mask(_block()).tolist() == [True, False, False, False]


def test_a_block_without_coordinates_raises():
    with pytest.raises(KeyError):
        mp.Cuboid.cube(2.0).mask(mp.Block({"type_id": np.array([1, 2])}))
