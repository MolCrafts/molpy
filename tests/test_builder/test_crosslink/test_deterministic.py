"""Tests for ``DeterministicCrosslinker`` (crosslink-01).

Exhaustive 100% pairing, uniform ``spacing`` (topological), reproducibility,
explicit ``pairs``, and the exclusion switches.
"""

import inspect

import molpy as mp
from molpy.builder.crosslink import DeterministicCrosslinker

RXN = "[N:1].[O:2]>>[N:1][O:2]"
# Single-atom self-pattern: every carbon is a site (used for spacing / A x A).
RXN_CC = "[C:1].[C:2]>>[C:1][C:2]"


def _carbon_chain(m):
    """Linear chain of ``m`` carbons: C0-C1-...-C(m-1) (no coordinates)."""
    s = mp.Atomistic()
    prev = None
    for _ in range(m):
        c = s.def_atom(element="C")
        if prev is not None:
            s.def_bond(prev, c, order=1.0)
        prev = c
    return s


def _typed_carbon_chain(types):
    """Linear carbon chain whose atoms carry optional ``type`` labels."""
    s = mp.Atomistic()
    prev = None
    for kind in types:
        c = s.def_atom(element="C")
        if kind is not None:
            c["type"] = kind
        if prev is not None:
            s.def_bond(prev, c, order=1.0)
        prev = c
    return s


def _n_o_cloud(n):
    """``n`` nitrogens and ``n`` oxygens, all mutually within a small cutoff."""
    s = mp.Atomistic()
    for i in range(n):
        s.def_atom(element="N", x=float(i), y=0.0, z=0.0)
        s.def_atom(element="O", x=float(i), y=1.0, z=0.0)
    return s


def _bond_count(graph):
    return len(list(graph.bonds))


# --------------------------------------------------------------------------
# ac-004 — exhaustive 100%, no RNG / conversion
# --------------------------------------------------------------------------


def test_exhaustive_consumes_every_site():
    n = 5
    g = _n_o_cloud(n)
    before = _bond_count(g)
    out = DeterministicCrosslinker(RXN, cutoff=2.0).apply(g)
    # Each N pairs with exactly one O -> n new crosslink bonds.
    assert _bond_count(out) - before == n


def test_no_conversion_or_seed_parameters():
    params = inspect.signature(DeterministicCrosslinker.__init__).parameters
    assert "conversion" not in params
    assert "seed" not in params


# --------------------------------------------------------------------------
# ac-005 — spacing: uniform crosslink points (pure topology, no coords)
# --------------------------------------------------------------------------


def _reacted_sites(graph, base_bonds):
    """Number of distinct atoms that gained a new (crosslink) bond."""
    reacted = set()
    for i, b in enumerate(graph.bonds):
        if i >= base_bonds:
            reacted.add(b.itom.handle)
            reacted.add(b.jtom.handle)
    return len(reacted)


def test_spacing_thins_sites_uniformly():
    m = 24
    base = _carbon_chain(m)
    base_bonds = _bond_count(base)

    counts = {}
    for k in (2, 4, 8):
        out = DeterministicCrosslinker(
            RXN_CC, spacing=k, exclude_same_match=True
        ).apply(base)
        counts[k] = _reacted_sites(out, base_bonds)

    # Regular sites ~ ceil(m / k); reacted sites <= that and shrink with k.
    assert counts[2] > counts[4] > counts[8]
    # Roughly m / k reacted sites (allow +/- 1 for the odd leftover).
    assert abs(counts[2] - m // 2) <= 2
    assert abs(counts[4] - m // 4) <= 2


def test_spacing_works_without_coordinates():
    base = _carbon_chain(12)
    out = DeterministicCrosslinker(RXN_CC, spacing=3, exclude_same_match=True).apply(
        base
    )
    assert _bond_count(out) > _bond_count(base)  # selected sites reacted


def test_spacing_uses_label_aware_sites():
    base = _typed_carbon_chain([None, "cx", None, "cx", None, "cx"])
    base_bonds = _bond_count(base)

    out = DeterministicCrosslinker(
        "[C;%cx:1].[C;%cx:2]>>[C:1][C:2]",
        spacing=2,
        exclude_same_match=True,
    ).apply(base)

    new_bonds = list(out.bonds)[base_bonds:]
    assert len(new_bonds) == 1
    assert {new_bonds[0].itom.get("type"), new_bonds[0].jtom.get("type")} == {"cx"}


# --------------------------------------------------------------------------
# ac-006 — determinism, explicit pairs, exclusions
# --------------------------------------------------------------------------


def test_deterministic_reproducible():
    g = _n_o_cloud(6)
    cl = DeterministicCrosslinker(RXN, cutoff=5.0)
    out1 = cl.apply(g)
    out2 = cl.apply(g)

    def bond_signature(graph):
        return sorted(
            tuple(sorted((b.itom.handle, b.jtom.handle))) for b in graph.bonds
        )

    assert bond_signature(out1) == bond_signature(out2)


def test_explicit_pairs_only_forms_named_bonds():
    # 3 N sites (index 0,1,2) and 3 O sites (index 0,1,2).
    g = _n_o_cloud(3)
    base = _bond_count(g)
    out = DeterministicCrosslinker(RXN, pairs=[(0, 2), (1, 0)]).apply(g)
    assert _bond_count(out) - base == 2


def test_explicit_pairs_use_label_aware_site_indices():
    base = _typed_carbon_chain([None, "cx", None, "cx"])
    base_bonds = _bond_count(base)

    out = DeterministicCrosslinker(
        "[C;%cx:1].[C;%cx:2]>>[C:1][C:2]",
        pairs=[(0, 1)],
    ).apply(base)

    new_bonds = list(out.bonds)[base_bonds:]
    assert len(new_bonds) == 1
    assert {new_bonds[0].itom.get("type"), new_bonds[0].jtom.get("type")} == {"cx"}


def test_exclude_same_molecule():
    # One molecule holding both an N and an O; no cross-molecule partner.
    s = mp.Atomistic()
    n = s.def_atom(element="N")
    o = s.def_atom(element="O")
    s.def_bond(n, o, order=1.0)  # already one molecule
    base = _bond_count(s)
    out = DeterministicCrosslinker(RXN, exclude_same_molecule=True).apply(s)
    assert _bond_count(out) == base  # nothing to pair across molecules


def test_exclude_same_match_prevents_self_pairing():
    # A x A on a 2-carbon molecule; without exclude the atom could self-bind.
    s = mp.Atomistic()
    c0 = s.def_atom(element="C")
    c1 = s.def_atom(element="C")
    s.def_bond(c0, c1, order=1.0)
    base = _bond_count(s)
    out = DeterministicCrosslinker(RXN_CC, exclude_same_match=True).apply(s)
    # c0 and c1 may bond to each other, but no atom bonds to itself.
    for b in out.bonds:
        assert b.itom.handle != b.jtom.handle
    assert _bond_count(out) >= base
