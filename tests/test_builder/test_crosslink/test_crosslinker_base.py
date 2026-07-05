"""Tests for the ``Crosslinker`` base class (crosslink-01, base + helpers).

Covers the immutable ``apply`` boundary, the molrs-only engine contract, and
``cutoff`` distance filtering with topological fallback / fail-fast.
"""

import inspect

import pytest

import molpy as mp
import molrs
from molpy.builder.crosslink import Candidate, Crosslinker, DeterministicCrosslinker

# N + O -> N-O bond: two single-atom components, no orientation duplicates.
RXN = "[N:1].[O:2]>>[N:1][O:2]"


def _graph(spec):
    """Build lone atoms from ``(element, (x, y, z) | None)`` tuples."""
    s = mp.Atomistic()
    for element, xyz in spec:
        if xyz is None:
            s.def_atom(element=element)
        else:
            s.def_atom(element=element, x=xyz[0], y=xyz[1], z=xyz[2])
    return s


def _new_bonds(graph):
    return [
        (b.itom.get("element"), b.jtom.get("element"), b.itom.get("x"), b.jtom.get("x"))
        for b in graph.bonds
    ]


# --------------------------------------------------------------------------
# ac-001 — immutable apply boundary (copy-once, same subclass out)
# --------------------------------------------------------------------------


def test_apply_does_not_mutate_input():
    g = _graph([("N", (0, 0, 0)), ("O", (1, 0, 0))])
    n_atoms_before = g.n_atoms
    n_bonds_before = len(list(g.bonds))

    DeterministicCrosslinker(RXN, cutoff=5.0).apply(g)

    assert g.n_atoms == n_atoms_before
    assert len(list(g.bonds)) == n_bonds_before  # input untouched


def test_apply_returns_new_object_same_type_with_new_bond():
    g = _graph([("N", (0, 0, 0)), ("O", (1, 0, 0))])
    out = DeterministicCrosslinker(RXN, cutoff=5.0).apply(g)

    assert out is not g
    assert type(out) is type(g)
    assert len(list(out.bonds)) == len(list(g.bonds)) + 1


def test_apply_preserves_subclass():
    class MyAtomistic(mp.Atomistic):
        pass

    g = MyAtomistic()
    g.def_atom(element="N", x=0.0, y=0.0, z=0.0)
    g.def_atom(element="O", x=1.0, y=0.0, z=0.0)

    out = DeterministicCrosslinker(RXN, cutoff=5.0).apply(g)
    assert type(out) is MyAtomistic


# --------------------------------------------------------------------------
# ac-002 — engine lives in molrs; molpy only orchestrates
# --------------------------------------------------------------------------


def test_base_is_abstract():
    with pytest.raises(TypeError):
        Crosslinker(RXN)  # type: ignore[abstract]


def test_reaction_is_molrs_reaction():
    cl = DeterministicCrosslinker(RXN)
    assert isinstance(cl._reaction, molrs.Reaction)


def test_select_is_the_only_abstract_hook():
    assert getattr(Crosslinker.select, "__isabstractmethod__", False)


# --------------------------------------------------------------------------
# ac-003 — cutoff distance filter (molrs NeighborList) + xyz optional
# --------------------------------------------------------------------------


def test_cutoff_only_bonds_pairs_within_range():
    # Two well-separated N/O pairs; cross pairs are far outside the cutoff.
    g = _graph(
        [
            ("N", (0.0, 0.0, 0.0)),
            ("O", (1.0, 0.0, 0.0)),  # 1.0 from first N
            ("N", (20.0, 0.0, 0.0)),
            ("O", (20.5, 0.0, 0.0)),  # 0.5 from second N
        ]
    )
    out = DeterministicCrosslinker(RXN, cutoff=3.0).apply(g)
    bonds = _new_bonds(out)
    assert len(bonds) == 2
    # Every bond joins atoms within 3.0 (never the 20-apart cross pairs).
    for _, _, xi, xj in bonds:
        assert abs(xi - xj) <= 3.0


def test_cutoff_leaves_isolated_site_unreacted():
    g = _graph(
        [
            ("N", (0.0, 0.0, 0.0)),
            ("O", (1.0, 0.0, 0.0)),
            ("N", (50.0, 0.0, 0.0)),  # no O within cutoff
        ]
    )
    out = DeterministicCrosslinker(RXN, cutoff=3.0).apply(g)
    assert len(_new_bonds(out)) == 1


def test_no_coords_topological_pairing():
    # No coordinates + no cutoff -> topological all-combinations pairing.
    g = _graph([("N", None), ("N", None), ("O", None), ("O", None)])
    out = DeterministicCrosslinker(RXN).apply(g)
    assert len(_new_bonds(out)) == 2  # 2 N + 2 O, each consumed once


def test_cutoff_without_coordinates_raises():
    g = _graph([("N", None), ("O", None)])
    with pytest.raises(ValueError):
        DeterministicCrosslinker(RXN, cutoff=3.0).apply(g)


# --------------------------------------------------------------------------
# Candidate shape
# --------------------------------------------------------------------------


def test_candidate_is_frozen_dataclass():
    fields = {f.name for f in inspect.signature(Candidate).parameters.values()}
    assert {"occ_a", "occ_b", "comp_a", "comp_b", "distance"} <= fields
