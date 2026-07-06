"""ac-002 (incremental-typify-03): ``Crosslinker(rxn, typifier=ff)`` retypes each
formed crosslink's affected region via a per-``apply`` ``RetypeCache``, writing
interior types back onto the returned graph. Without a typifier the crosslinker
stays pure-topology (the graph is returned untyped) — unchanged behaviour.
"""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.builder.crosslink import DeterministicCrosslinker
from molpy.io.forcefield.xml import read_oplsaa_forcefield
from molpy.typifier import OplsTypifier

# Two CH3-CH2* radicals: the degree-3 CH2 carbons couple into butane (a
# valence-correct product the OPLS atom typifier can fully assign).
RXN = "[C;X3:1].[C;X3:2]>>[C:1][C:2]"


@pytest.fixture(scope="module")
def opls() -> OplsTypifier:
    ff = read_oplsaa_forcefield("oplsaa.xml")
    return OplsTypifier(ff, strict_typing=True)


def _ethyl_radical() -> mp.Atomistic:
    """CH3-CH2* — the terminal carbon has degree 3 (2 H + 1 C)."""
    s = mp.Atomistic()
    c1 = s.def_atom(element="C", symbol="C")  # CH3
    c2 = s.def_atom(element="C", symbol="C")  # CH2 (reactive)
    s.def_bond(c1, c2)
    for _ in range(3):
        s.def_bond(c1, s.def_atom(element="H", symbol="H"))
    for _ in range(2):
        s.def_bond(c2, s.def_atom(element="H", symbol="H"))
    return s


def _two_radicals() -> mp.Atomistic:
    g = mp.Atomistic()
    g.merge(_ethyl_radical())
    g.merge(_ethyl_radical())
    return g


# --------------------------------------------------------------------------
# default (no typifier) — pure topology, graph untyped
# --------------------------------------------------------------------------
def test_crosslink_without_typifier_leaves_graph_untyped() -> None:
    g = _two_radicals()
    base_bonds = len(list(g.bonds))
    out = DeterministicCrosslinker(RXN, exclude_same_molecule=True).apply(g)

    assert len(list(out.bonds)) == base_bonds + 1  # one crosslink formed
    assert all(a.get("type") is None for a in out.atoms)  # unchanged behaviour


# --------------------------------------------------------------------------
# with typifier — interior retyped, matching the whole-graph baseline
# --------------------------------------------------------------------------
def test_crosslink_with_typifier_types_interior(opls: OplsTypifier) -> None:
    g = _two_radicals()
    xlink = DeterministicCrosslinker(RXN, exclude_same_molecule=True, typifier=opls)
    out = xlink.apply(g)

    assert len(xlink.last_regions) == 1
    region = xlink.last_regions[0]

    # The two joined carbons (the region interior) carry OPLS types.
    interior_parents = [region.entity_map[a] for a in region.interior]
    assert len(interior_parents) >= 2
    for atom in interior_parents:
        assert atom.get("type") is not None

    # Every atom that received a type matches the whole-graph OPLS baseline.
    baseline = opls.typify(out.get_topo(gen_angle=True, gen_dihe=True))
    baseline_by_handle = {
        pa.handle: b.get("type") for pa, b in zip(out.atoms, baseline.atoms)
    }
    typed = 0
    for atom in out.atoms:
        t = atom.get("type")
        if t is not None:
            assert t == baseline_by_handle[atom.handle]
            typed += 1
    assert typed >= 2


def test_crosslink_typifier_does_not_mutate_input(opls: OplsTypifier) -> None:
    g = _two_radicals()
    DeterministicCrosslinker(RXN, exclude_same_molecule=True, typifier=opls).apply(g)
    # Input graph is copied by ``apply``; retyping must not leak onto it.
    assert all(a.get("type") is None for a in g.atoms)
