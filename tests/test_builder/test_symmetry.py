"""Tests for native space-group expansion (molpy.builder.symmetry)."""

import numpy as np
import pytest

from molpy.builder import Lattice, Site, SpaceGroup, build_crystal
from molpy.builder.symmetry import parse_triplet

# Full Ia-3d (No. 230) operator list, as a CIF would list it — the regression
# anchor: cubic garnet Li7La3Zr2O12 is built from exactly these 96 operators.
IA3D = """x,y,z;1/4-y,3/4+x,1/4+z;-x,1/2-y,z;1/4+y,1/4-x,3/4+z;1/4+x,1/4-z,3/4+y;
x,-y,1/2-z;3/4+x,1/4+z,1/4-y;3/4+z,1/4+y,1/4-x;1/2-x,y,-z;1/4-z,3/4+y,1/4+x;
z,x,y;y,z,x;-y,1/2-z,x;z,-x,1/2-y;1/2-y,z,-x;-z,1/2-x,y;1/2-z,x,-y;y,-z,1/2-x;
3/4+y,1/4+x,1/4-z;1/4-y,1/4-x,1/4-z;1/4-x,3/4+z,1/4+y;1/4-x,1/4-z,1/4-y;
1/4+z,1/4-y,3/4+x;1/4-z,1/4-y,1/4-x;-x,-y,-z;1/4+y,3/4-x,1/4-z;x,1/2+y,-z;
1/4-y,1/4+x,3/4-z;1/4-x,1/4+z,3/4-y;-x,y,1/2+z;3/4-x,1/4-z,1/4+y;3/4-z,1/4-y,1/4+x;
1/2+x,-y,z;1/4+z,3/4-y,1/4-x;-z,-x,-y;-y,-z,-x;y,1/2+z,-x;-z,x,1/2+y;1/2+y,-z,x;
z,1/2+x,-y;1/2+z,-x,y;-y,z,1/2+x;3/4-y,1/4-x,1/4+z;1/4+y,1/4+x,1/4+z;
1/4+x,3/4-z,1/4-y;1/4+x,1/4+z,1/4+y;1/4-z,1/4+y,3/4-x;1/4+z,1/4+y,1/4+x;
1/2+x,1/2+y,1/2+z;3/4-y,5/4+x,3/4+z;1/2-x,1-y,1/2+z;3/4+y,3/4-x,5/4+z;
3/4+x,3/4-z,5/4+y;1/2+x,1/2-y,1-z;5/4+x,3/4+z,3/4-y;5/4+z,3/4+y,3/4-x;
1-x,1/2+y,1/2-z;3/4-z,5/4+y,3/4+x;1/2+z,1/2+x,1/2+y;1/2+y,1/2+z,1/2+x;
1/2-y,1-z,1/2+x;1/2+z,1/2-x,1-y;1-y,1/2+z,1/2-x;1/2-z,1-x,1/2+y;1-z,1/2+x,1/2-y;
1/2+y,1/2-z,1-x;5/4+y,3/4+x,3/4-z;3/4-y,3/4-x,3/4-z;3/4-x,5/4+z,3/4+y;
3/4-x,3/4-z,3/4-y;3/4+z,3/4-y,5/4+x;3/4-z,3/4-y,3/4-x;1/2-x,1/2-y,1/2-z;
3/4+y,5/4-x,3/4-z;1/2+x,1+y,1/2-z;3/4-y,3/4+x,5/4-z;3/4-x,3/4+z,5/4-y;
1/2-x,1/2+y,1+z;5/4-x,3/4-z,3/4+y;5/4-z,3/4-y,3/4+x;1+x,1/2-y,1/2+z;
3/4+z,5/4-y,3/4-x;1/2-z,1/2-x,1/2-y;1/2-y,1/2-z,1/2-x;1/2+y,1+z,1/2-x;
1/2-z,1/2+x,1+y;1+y,1/2-z,1/2+x;1/2+z,1+x,1/2-y;1+z,1/2-x,1/2+y;1/2-y,1/2+z,1+x;
5/4-y,3/4-x,3/4+z;3/4+y,3/4+x,3/4+z;3/4+x,5/4-z,3/4-y;3/4+x,3/4+z,3/4+y;
3/4-z,3/4+y,5/4-x;3/4+z,3/4+y,3/4+x"""
IA3D_OPS = [op.strip() for op in IA3D.replace("\n", "").split(";") if op.strip()]


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


def test_ia3d_is_a_closed_group_of_order_96():
    sg = SpaceGroup.from_triplets(IA3D_OPS)
    assert sg.order == 96


def test_generators_close_to_full_group():
    # A 4-fold screw, the 3-fold, inversion, and the I-centering generate all 96.
    gens = [IA3D_OPS[1], IA3D_OPS[10], IA3D_OPS[24], IA3D_OPS[48]]
    sg = SpaceGroup.from_generators(gens)
    assert sg.order == 96


@pytest.mark.parametrize(
    "frac, multiplicity",
    [
        ((0.0, 0.25, 0.125), 24),  # La 24c
        ((0.0, 0.0, 0.0), 16),  # Zr 16a
        ((-0.03161, 0.05454, 0.14940), 96),  # O 96h (general position)
        ((-0.125, 0.0, 0.25), 24),  # Li 24d
    ],
)
def test_llzo_site_multiplicities(frac, multiplicity):
    sg = SpaceGroup.from_triplets(IA3D_OPS)
    images = sg.equivalent_positions(frac)
    assert len(images) == multiplicity


def test_from_spacegroup_builds_full_garnet_cell():
    a = 12.9727
    sg = SpaceGroup.from_triplets(IA3D_OPS)
    sites = [
        Site("La", "La", (0.0, 0.25, 0.125)),
        Site("Zr", "Zr", (0.0, 0.0, 0.0)),
        Site("O", "O", (-0.03161, 0.05454, 0.14940)),
    ]
    lat = Lattice.from_spacegroup(a * np.eye(3), sites, sg)
    # 24 La + 16 Zr + 96 O per conventional cell.
    assert len(lat.basis) == 24 + 16 + 96
    cell = build_crystal(lat, repeats=(1, 1, 1))
    assert len(list(cell.atoms)) == 136
