"""Tests for ``PortMatcher`` (crosslink-03).

Reads modelled ``atom["port"]`` markers into molrs-shaped occurrences and proves
they can drive a ``Crosslinker`` through the same molrs edit path.
"""

import molrs

import molpy as mp
from molpy.builder.crosslink import DeterministicCrosslinker, PortMatcher
from molpy.parser import parse_monomer


def _ported(spec):
    """Build lone carbons carrying ``port`` markers from ``(port | None)`` list."""
    s = mp.Atomistic()
    for port in spec:
        if port is None:
            s.def_atom(element="C")
        else:
            s.def_atom(element="C", port=port)
    return s


# --------------------------------------------------------------------------
# ac-001 — port markers -> molrs-shaped occurrences
# --------------------------------------------------------------------------


def test_port_matcher_returns_molrs_shaped_occurrences():
    g = _ported([">", "<", ">", None])
    occ = PortMatcher().find_matches_mapped(g, ">")

    # list[dict[int, int]] with map number 1 = the port atom, same shape as
    # molrs.SmartsPattern.find_matches_mapped.
    assert isinstance(occ, list)
    assert all(set(o) == {1} for o in occ)
    assert len(occ) == 2  # two ">" ports
    for o in occ:
        assert g.get(o[1], "port") == ">"


def test_port_matcher_shape_matches_molrs():
    g = mp.Atomistic()
    g.def_atom(element="C", x=0.0, y=0.0, z=0.0, port=">")
    port_occ = PortMatcher().find_matches_mapped(g, ">")
    smarts_occ = molrs.SmartsPattern("[C:1]").find_matches_mapped(g)
    # Same container/element structure: list of {int: int} single-atom maps.
    assert type(port_occ) is type(smarts_occ)
    assert {frozenset(o) for o in port_occ} == {frozenset(o) for o in smarts_occ}


def test_port_key_is_configurable():
    g = mp.Atomistic()
    g.def_atom(element="C", site="head")
    occ = PortMatcher(port_key="site").find_matches_mapped(g, "head")
    assert len(occ) == 1


def test_reads_bigsmiles_descriptor_ports():
    # Real BigSMILES path: {[<]CCO[>]} marks atom["port"] = "<" / ">".
    monomer = parse_monomer("{[<]CCO[>]}")
    right = PortMatcher().find_matches_mapped(monomer, ">")
    left = PortMatcher().find_matches_mapped(monomer, "<")
    assert len(right) == 1
    assert len(left) == 1
    assert monomer.get(right[0][1], "port") == ">"


# --------------------------------------------------------------------------
# ac-002 — PortMatcher occurrences consumed by a Crosslinker
# --------------------------------------------------------------------------


class _PortCrosslinker(DeterministicCrosslinker):
    """Crosslinker that selects sites from ``port`` markers instead of SMARTS."""

    def __init__(self, reaction, *, port_a, port_b, **kwargs):
        super().__init__(reaction, **kwargs)
        self._matcher = PortMatcher()
        self._port_a = port_a
        self._port_b = port_b

    def _match_occurrences(self, graph):
        occurrences = [[] for _ in self._reaction.reactant_patterns]
        occurrences[self._comp_a] = [
            {self._map_a: o[1]}
            for o in self._matcher.find_matches_mapped(graph, self._port_a)
        ]
        occurrences[self._comp_b] = [
            {self._map_b: o[1]}
            for o in self._matcher.find_matches_mapped(graph, self._port_b)
        ]
        return occurrences


def test_port_occurrences_drive_crosslink():
    # Two separate one-carbon "monomers": one head port ">", one tail port "<".
    g = mp.Atomistic()
    g.def_atom(element="C", port=">")
    g.def_atom(element="C", port="<")
    base = len(list(g.bonds))

    out = _PortCrosslinker("[C:1].[C:2]>>[C:1][C:2]", port_a=">", port_b="<").apply(g)

    new_bonds = list(out.bonds)
    assert len(new_bonds) == base + 1  # the ">" carbon bonded to the "<" carbon
    joined = {new_bonds[0].itom.get("port"), new_bonds[0].jtom.get("port")}
    assert joined == {">", "<"}
