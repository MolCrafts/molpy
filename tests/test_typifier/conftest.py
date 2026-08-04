"""Shared fixtures for the typifier suite.

Heavy force-field typifiers (CL&P / OPLS-AA / MMFF) parse large XML and build
``ForceFieldParams`` indexes. Construct each **once per module** that needs it;
tests only call ``.typify`` / ``.apply``.
"""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.core.atomistic import Atomistic
from molpy.typifier import ClpTypifier, MMFFTypifier


def _build(elements: list[str], edges: list[tuple[int, int]]) -> Atomistic:
    asm = Atomistic()
    atoms = [asm.def_atom(element=e) for e in elements]
    for i, j in edges:
        asm.def_bond(atoms[i], atoms[j])
    return asm.get_topo(gen_angle=True, gen_dihe=True)


def c4c1im_graph() -> Atomistic:
    """[C4C1im]+ connectivity (paduagroup/clandp z-matrix)."""
    el = [
        "N",
        "C",
        "N",
        "C",
        "C",
        "C",
        "H",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "C",
        "H",
        "H",
        "H",
        "H",
        "H",
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 5),
        (1, 6),
        (2, 7),
        (3, 8),
        (4, 9),
        (5, 10),
        (5, 11),
        (5, 12),
        (7, 13),
        (7, 14),
        (7, 15),
        (13, 16),
        (13, 17),
        (13, 18),
        (16, 19),
        (16, 20),
        (16, 21),
        (19, 22),
        (19, 23),
        (19, 24),
    ]
    return _build(el, edges)


def bf4_graph() -> Atomistic:
    return _build(["B", "F", "F", "F", "F"], [(0, 1), (0, 2), (0, 3), (0, 4)])


@pytest.fixture(scope="module")
def clp() -> ClpTypifier:
    """One non-strict CL&P typifier for the whole module."""
    return ClpTypifier(strict=False)


@pytest.fixture(scope="module")
def c4c1im_typed(clp: ClpTypifier) -> Atomistic:
    """[C4C1im]+ fully CL&P-typed once; Drude / charge tests reuse this graph."""
    return clp.typify(c4c1im_graph())


@pytest.fixture(scope="module")
def mmff() -> MMFFTypifier:
    return MMFFTypifier()
