"""Unit tests for :mod:`molpy.builder.assembly._residue_graph`."""

import pytest

from molpy.builder.assembly import linear_cgsmiles, ring_cgsmiles, star_cgsmiles


class TestLinearCgsmiles:
    def test_homopolymer_uses_repeat_syntax(self):
        assert linear_cgsmiles(["EO"] * 5) == "{[#EO]|5}"

    def test_sequence_preserves_label_order(self):
        assert linear_cgsmiles(["A", "B", "A"]) == "{[#A][#B][#A]}"

    def test_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            linear_cgsmiles([])


class TestRingCgsmiles:
    def test_places_ring_digit_on_first_and_last_nodes(self):
        assert ring_cgsmiles("EO", 3) == "{[#EO]1[#EO][#EO]1}"

    def test_ring_needs_three_residues(self):
        with pytest.raises(ValueError, match="n >= 3"):
            ring_cgsmiles("EO", 2)


class TestStarCgsmiles:
    def test_formats_branches_and_main_arm(self):
        assert star_cgsmiles("X3", "EO", n_arms=3, arm_length=2) == (
            "{[#X3]([#EO]|2)([#EO]|2)[#EO]|2}"
        )

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"n_arms": 1, "arm_length": 2}, "n_arms >= 2"),
            ({"n_arms": 2, "arm_length": 0}, "arm_length must be >= 1"),
        ],
    )
    def test_invalid_star_shape_is_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            star_cgsmiles("X", "A", **kwargs)
