"""Tests for ``new random(...)`` weighted-sampler expansion."""

from __future__ import annotations

from molpy.parser.moltemplate import build_system, parse_string


_SRC_COMMON = """
Methyl {
  write('Data Atoms') {
    $atom:c $mol:m @atom:C 0.0 0.0 0.0 0.0
  }
}

Ethyl {
  write('Data Atoms') {
    $atom:c1 $mol:m @atom:C 0.0 0.0 0.0 0.0
    $atom:c2 $mol:m @atom:C 0.0 1.5 0.0 0.0
  }
}
"""


def test_random_exact_counts_sum_to_grid():
    src = (
        _SRC_COMMON
        + """
mix = new random([Methyl, Ethyl], [4, 6]) [10].move(5, 0, 0)
"""
    )
    system, _ = build_system(parse_string(src))
    # Exact counts: 4 Methyl (1 atom each) + 6 Ethyl (2 atoms) = 4 + 12 = 16
    assert len(list(system.atoms)) == 16


def test_random_weighted_seeded_is_deterministic():
    src = (
        _SRC_COMMON
        + """
mix = new random([Methyl, Ethyl], [0.5, 0.5], 42) [10].move(5, 0, 0)
"""
    )
    a, _ = build_system(parse_string(src))
    b, _ = build_system(parse_string(src))
    # Same seed → same atom count (and same per-instance composition).
    assert len(list(a.atoms)) == len(list(b.atoms))


def test_random_uniform_without_weights():
    # No weights list → uniform probability.
    src = (
        _SRC_COMMON
        + """
mix = new random([Methyl, Ethyl]) [5].move(5, 0, 0)
"""
    )
    system, _ = build_system(parse_string(src))
    # Exactly 5 molecules; each is Methyl (1 atom) or Ethyl (2 atoms)
    n_atoms = len(list(system.atoms))
    assert 5 <= n_atoms <= 10


def test_random_preserves_post_class_transforms():
    # Per-choice transforms inside the list should survive.
    src = (
        _SRC_COMMON
        + """
mix = new random([Methyl.move(100, 0, 0), Ethyl.move(0, 100, 0)], [5, 0]) [5].move(0, 0, 0)
"""
    )
    system, _ = build_system(parse_string(src))
    xs = [float(a.get("x", 0.0)) for a in system.atoms]
    # All instances should be Methyl (weight 5 vs 0) → x ≈ 100
    assert all(abs(x - 100.0) < 1e-6 for x in xs), xs
