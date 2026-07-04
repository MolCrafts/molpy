"""Unit tests for the deterministic copolymer sequence generators."""

import numpy as np
import pytest

from molpy.builder.polymer.sequences import (
    AlternatingSequenceGenerator,
    BlockSequenceGenerator,
    SequenceGenerator,
)


class TestAlternatingSequenceGenerator:
    def test_two_monomers_alternate(self):
        seq = AlternatingSequenceGenerator(["A", "B"]).generate_sequence(6)
        assert seq == ["A", "B", "A", "B", "A", "B"]

    def test_three_monomers_cycle(self):
        seq = AlternatingSequenceGenerator(["A", "B", "C"]).generate_sequence(7)
        assert seq == ["A", "B", "C", "A", "B", "C", "A"]

    def test_odd_dp_exact_counts(self):
        # 2*mil ions -> exactly mil of each, strictly alternating
        seq = AlternatingSequenceGenerator(["C", "A"]).generate_sequence(8)
        assert seq.count("C") == 4 and seq.count("A") == 4
        assert all(seq[i] != seq[i + 1] for i in range(len(seq) - 1))

    def test_dp_zero(self):
        assert AlternatingSequenceGenerator(["A", "B"]).generate_sequence(0) == []

    def test_rng_is_ignored(self):
        gen = AlternatingSequenceGenerator(["A", "B"])
        a = gen.generate_sequence(5, rng=np.random.default_rng(1))
        b = gen.generate_sequence(5, rng=np.random.default_rng(999))
        assert a == b == ["A", "B", "A", "B", "A"]

    def test_expected_composition(self):
        comp = AlternatingSequenceGenerator(["A", "B", "C"]).expected_composition()
        assert comp == pytest.approx({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})
        assert sum(comp.values()) == pytest.approx(1.0)

    def test_requires_two_ids(self):
        with pytest.raises(ValueError, match="at least 2"):
            AlternatingSequenceGenerator(["A"])

    def test_rejects_empty_id(self):
        with pytest.raises(ValueError, match="non-empty strings"):
            AlternatingSequenceGenerator(["A", ""])

    def test_is_a_sequence_generator(self):
        gen: SequenceGenerator = AlternatingSequenceGenerator(["A", "B"])
        assert len(gen.generate_sequence(3, np.random.default_rng(0))) == 3


class TestBlockSequenceGenerator:
    def test_two_equal_blocks(self):
        seq = BlockSequenceGenerator({"A": 0.5, "B": 0.5}).generate_sequence(6)
        assert seq == ["A", "A", "A", "B", "B", "B"]

    def test_exact_integer_counts_via_largest_remainder(self):
        # fractions 8:8:2 of dp=18 -> exactly 8, 8, 2 in block order
        seq = BlockSequenceGenerator({"C": 8, "A": 8, "P": 2}).generate_sequence(18)
        assert seq == ["C"] * 8 + ["A"] * 8 + ["P"] * 2

    def test_remainder_distribution_sums_to_dp(self):
        # 1:1:1 of dp=10 -> 4,3,3 (largest remainders first), total 10
        seq = BlockSequenceGenerator({"A": 1, "B": 1, "C": 1}).generate_sequence(10)
        assert len(seq) == 10
        assert seq == ["A"] * 4 + ["B"] * 3 + ["C"] * 3
        # blocks are contiguous
        assert seq[:4] == ["A"] * 4

    def test_block_order_follows_mapping(self):
        seq = BlockSequenceGenerator({"B": 0.5, "A": 0.5}).generate_sequence(4)
        assert seq == ["B", "B", "A", "A"]

    def test_dp_zero(self):
        assert BlockSequenceGenerator({"A": 1.0}).generate_sequence(0) == []

    def test_expected_composition_normalised(self):
        comp = BlockSequenceGenerator({"A": 3, "B": 1}).expected_composition()
        assert comp == pytest.approx({"A": 0.75, "B": 0.25})

    def test_rejects_empty_mapping(self):
        with pytest.raises(ValueError, match="non-empty"):
            BlockSequenceGenerator({})

    def test_rejects_nonpositive_total(self):
        with pytest.raises(ValueError, match="positive"):
            BlockSequenceGenerator({"A": 0.0})

    def test_rejects_negative_fraction(self):
        with pytest.raises(ValueError, match="non-negative"):
            BlockSequenceGenerator({"A": 1.0, "B": -0.5})

    def test_is_a_sequence_generator(self):
        gen: SequenceGenerator = BlockSequenceGenerator({"A": 0.5, "B": 0.5})
        assert len(gen.generate_sequence(4, np.random.default_rng(0))) == 4
