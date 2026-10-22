"""Space-group expansion for the crystal builder (``mp.builder.SpaceGroup``)."""

import numpy as np
import pytest

from molpy.builder.symmetry import parse_triplet
import molpy as mp

#: Generators of Fm-3m (No. 225): the two F-centring translations, then 2z, 2y,
#: 3[111], 2[110] and the inversion, which generate m-3m (order 48).
FM3M_GENERATORS = [
    "x+1/2,y+1/2,z",
    "x+1/2,y,z+1/2",
    "-x,-y,z",
    "-x,y,-z",
    "z,x,y",
    "y,x,-z",
    "-x,-y,-z",
]


class TestSpaceGroup:
    def test_identity_generator_has_order_one(self):
        group = mp.builder.SpaceGroup.from_generators(["x,y,z"])
        assert group.order == 1

    def test_fm3m_closes_to_order_192(self):
        assert mp.builder.SpaceGroup.from_generators(FM3M_GENERATORS).order == 192

    def test_r3_closes_to_order_9(self):
        """R3 (hexagonal axes): a 3-fold axis and the (2/3, 1/3, 1/3) centring.

        Composing 1/3 and 2/3 gives 0.9999999999999999, which must key as the
        lattice translation 0, not as a new operator.
        """
        group = mp.builder.SpaceGroup.from_generators(["-y,x-y,z", "x+2/3,y+1/3,z+1/3"])
        assert group.order == 9

    def test_six_fold_screw_closes_to_order_6(self):
        assert mp.builder.SpaceGroup.from_generators(["-x+y,-x,z+1/6"]).order == 6

    def test_closed_translations_lie_in_unit_interval(self):
        group = mp.builder.SpaceGroup.from_generators(["-y,x-y,z", "x+2/3,y+1/3,z+1/3"])
        for _, t in group.operators:
            assert np.all((t >= 0.0) & (t < 1.0))

    def test_from_triplets_accepts_cif_meta_tuple(self):
        group = mp.builder.SpaceGroup.from_triplets(("x,y,z", "-x,-y,-z"))
        assert group.order == 2


@pytest.fixture(scope="module")
def fm3m():
    return mp.builder.SpaceGroup.from_generators(FM3M_GENERATORS)


class TestEquivalentPositions:
    def test_general_position_has_full_multiplicity(self, fm3m):
        assert len(fm3m.equivalent_positions([0.13, 0.27, 0.41])) == 192

    @pytest.mark.parametrize(
        ("site", "multiplicity"),
        [
            ((0.0, 0.0, 0.0), 4),  # 4a
            ((0.5, 0.5, 0.5), 4),  # 4b
            ((0.25, 0.25, 0.25), 8),  # 8c
            ((0.2, 0.0, 0.0), 24),  # 24e
        ],
    )
    def test_special_position_collapses(self, fm3m, site, multiplicity):
        images = fm3m.equivalent_positions(site)
        assert len(images) == multiplicity
        assert np.all((images >= 0.0) & (images < 1.0))

    def test_images_within_symprec_of_one_wrap_to_zero(self):
        group = mp.builder.SpaceGroup.from_triplets(["x,y,z", "x+2/3,y+1/3,z+1/3"])
        images = group.equivalent_positions([1 / 3, 2 / 3, 2 / 3])
        assert len(images) == 2
        assert np.all(images < 1.0)
        assert any(np.allclose(p, 0.0) for p in images)


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
