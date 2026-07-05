"""Tests for ``RandomCrosslinker`` (crosslink-02).

Conversion-targeted random network growth with a reproducible ``seed``. Only
``select`` is overridden; matching / distance / graph edit stay in the base.
"""

import molpy as mp
from molpy.builder.crosslink import Crosslinker, RandomCrosslinker

RXN = "[N:1].[O:2]>>[N:1][O:2]"


def _n_o_cloud(n_n, n_o, spacing=1.0):
    """``n_n`` lone nitrogens and ``n_o`` lone oxygens on a compact lattice."""
    s = mp.Atomistic()
    for i in range(n_n):
        s.def_atom(element="N", x=float(i) * spacing, y=0.0, z=0.0)
    for i in range(n_o):
        s.def_atom(element="O", x=float(i) * spacing, y=1.0, z=0.0)
    return s


def _bond_count(graph):
    return len(list(graph.bonds))


def _new_bonds(graph, base):
    return _bond_count(graph) - base


def _bond_signature(graph):
    return sorted(tuple(sorted((b.itom.handle, b.jtom.handle))) for b in graph.bonds)


# --------------------------------------------------------------------------
# ac-001 — conversion control
# --------------------------------------------------------------------------


def test_conversion_half_reacts_half_of_limiting_reactant():
    g = _n_o_cloud(10, 10)
    base = _bond_count(g)
    out = RandomCrosslinker(RXN, conversion=0.5, seed=1).apply(g)
    # limiting reactant = 10 sites; 0.5 conversion -> ~5 reactions (+/- 1 step).
    assert abs(_new_bonds(out, base) - 5) <= 1


def test_full_conversion_reacts_all_of_limiting_reactant():
    g = _n_o_cloud(6, 6)
    base = _bond_count(g)
    out = RandomCrosslinker(RXN, conversion=1.0, seed=3).apply(g)
    assert _new_bonds(out, base) == 6


# --------------------------------------------------------------------------
# ac-002 — seed reproducibility
# --------------------------------------------------------------------------


def test_same_seed_reproduces_exact_network():
    g = _n_o_cloud(8, 8)
    cl = RandomCrosslinker(RXN, conversion=0.5, seed=42)
    assert _bond_signature(cl.apply(g)) == _bond_signature(cl.apply(g))


def test_different_seeds_can_differ():
    g = _n_o_cloud(8, 8)
    a = RandomCrosslinker(RXN, conversion=0.5, seed=1).apply(g)
    b = RandomCrosslinker(RXN, conversion=0.5, seed=2).apply(g)
    # Not a hard guarantee for every pair, but seeds 1 vs 2 here diverge.
    assert _bond_signature(a) != _bond_signature(b)


# --------------------------------------------------------------------------
# ac-003 — conversion=1.0 terminates when candidates run out
# --------------------------------------------------------------------------


def test_full_conversion_terminates_when_unpairable():
    # Both reactive atoms sit on the SAME molecule; exclude_same_molecule leaves
    # no valid pairing -> must terminate (loop is bounded), not hang.
    s = mp.Atomistic()
    n = s.def_atom(element="N")
    o = s.def_atom(element="O")
    s.def_bond(n, o, order=1.0)
    base = _bond_count(s)
    out = RandomCrosslinker(
        RXN, conversion=1.0, seed=0, exclude_same_molecule=True
    ).apply(s)
    assert _new_bonds(out, base) == 0


# --------------------------------------------------------------------------
# ac-004 — cutoff / exclude / max_per_molecule
# --------------------------------------------------------------------------


def test_cutoff_limits_random_pairing_to_neighbors():
    g = mp.Atomistic()
    g.def_atom(element="N", x=0.0, y=0.0, z=0.0)
    g.def_atom(element="O", x=1.0, y=0.0, z=0.0)  # within cutoff of first N
    g.def_atom(element="N", x=30.0, y=0.0, z=0.0)  # isolated
    base = _bond_count(g)
    out = RandomCrosslinker(RXN, conversion=1.0, seed=5, cutoff=3.0).apply(g)
    assert _new_bonds(out, base) == 1


def test_max_per_molecule_caps_consumed_sites():
    # One molecule holding 3 nitrogens (3 sites) + 5 lone oxygens.
    s = mp.Atomistic()
    prev = None
    n_handles = []
    for _ in range(3):
        n = s.def_atom(element="N")
        n_handles.append(n.handle)
        if prev is not None:
            s.def_bond(prev, n, order=1.0)
        prev = n
    for _ in range(5):
        s.def_atom(element="O")
    base = _bond_count(s)
    out = RandomCrosslinker(RXN, conversion=1.0, seed=7, max_per_molecule=2).apply(s)
    # The 3-nitrogen molecule may consume at most 2 of its sites.
    consumed_n = 0
    for i, b in enumerate(out.bonds):
        if i < base:
            continue
        for h in (b.itom.handle, b.jtom.handle):
            if h in n_handles:
                consumed_n += 1
    assert consumed_n <= 2


def test_exclude_same_match_prevents_self_pair():
    # A x A single-atom reaction on a 2-carbon molecule.
    s = mp.Atomistic()
    c0 = s.def_atom(element="C")
    c1 = s.def_atom(element="C")
    s.def_bond(c0, c1, order=1.0)
    out = RandomCrosslinker(
        "[C:1].[C:2]>>[C:1][C:2]", conversion=1.0, seed=2, exclude_same_match=True
    ).apply(s)
    for b in out.bonds:
        assert b.itom.handle != b.jtom.handle


# --------------------------------------------------------------------------
# ac-005 — only select is overridden; base owns match/distance/edit
# --------------------------------------------------------------------------


def test_random_only_overrides_select():
    assert issubclass(RandomCrosslinker, Crosslinker)
    own = set(RandomCrosslinker.__dict__)
    assert "select" in own
    # Matching / candidate construction / graph edit are inherited, not redefined.
    assert "apply" not in own
    assert "_candidate_pairs" not in own


def test_immutable_inherited():
    g = _n_o_cloud(4, 4)
    n_atoms = g.n_atoms
    n_bonds = _bond_count(g)
    out = RandomCrosslinker(RXN, conversion=0.5, seed=1).apply(g)
    assert out is not g
    assert type(out) is type(g)
    assert g.n_atoms == n_atoms
    assert _bond_count(g) == n_bonds
