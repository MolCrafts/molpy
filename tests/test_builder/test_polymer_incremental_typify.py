"""ac-001 (incremental-typify-03): a shared ``RetypeCache`` threaded through
``PolymerBuilder.build`` dedups structurally identical junctions across the whole
chain, so the underlying atom typifier is invoked a BOUNDED number of times
(≈ #distinct junction environments) — flat as the chain length ``N`` grows, not
O(N). The typed product matches a whole-chain ``typify`` baseline.
"""

from __future__ import annotations

import pytest

from molpy.builder.polymer import Connector, PolymerBuilder
from molpy.core.atomistic import Atomistic
from molpy.io.forcefield.xml import read_oplsaa_forcefield
from molpy.reacter import Reacter, form_single_bond, select_hydrogens, select_self
from molpy.typifier import OplsTypifier


@pytest.fixture(scope="module")
def opls_ff():
    return read_oplsaa_forcefield("oplsaa.xml")


def _ethane_monomer() -> Atomistic:
    """CH3-CH3 with ``<`` / ``>`` ports on its two carbons."""
    s = Atomistic()
    c1 = s.def_atom(element="C", symbol="C", port="<")
    c2 = s.def_atom(element="C", symbol="C", port=">")
    s.def_bond(c1, c2)
    for c in (c1, c2):
        for _ in range(3):
            s.def_bond(c, s.def_atom(element="H", symbol="H"))
    return s


def _coupling_reacter() -> Reacter:
    """C-C coupling that drops one H from each anchor."""
    return Reacter(
        name="cc_coupling",
        anchor_selector_left=select_self,
        anchor_selector_right=select_self,
        leaving_selector_left=select_hydrogens(1),
        leaving_selector_right=select_hydrogens(1),
        bond_former=form_single_bond,
    )


class _CallCounter:
    """Wrap ``obj.method`` with an invocation counter (restore via ``stop``)."""

    def __init__(self, obj: object, method: str) -> None:
        self.count = 0
        self._obj = obj
        self._method = method
        self._orig = getattr(obj, method)

        def wrapper(*args: object, **kwargs: object) -> object:
            self.count += 1
            return self._orig(*args, **kwargs)

        setattr(obj, method, wrapper)

    def stop(self) -> None:
        setattr(self._obj, self._method, self._orig)


def _build_chain(n: int, typifier: OplsTypifier | None) -> Atomistic:
    connector = Connector(
        reacter=_coupling_reacter(), port_map={("E", "E"): (">", "<")}
    )
    builder = PolymerBuilder(
        library={"E": _ethane_monomer()}, connector=connector, typifier=typifier
    )
    return builder.build(f"{{[#E]|{n}}}").polymer


def _typify_calls(n: int, opls_ff) -> int:
    opls = OplsTypifier(opls_ff, strict_typing=True)
    spy = _CallCounter(opls.atom_typifier, "typify")
    try:
        _build_chain(n, opls)
    finally:
        spy.stop()
    return spy.count


# --------------------------------------------------------------------------
# ac-001 — bounded / flat junction typing cost
# --------------------------------------------------------------------------
def test_junction_typify_calls_flat_as_chain_grows(opls_ff) -> None:
    calls_8 = _typify_calls(8, opls_ff)
    calls_16 = _typify_calls(16, opls_ff)

    # Bounded by the (constant) number of distinct junction environments, not
    # the chain length: the two counts must be equal.
    assert calls_8 == calls_16, (
        f"junction typing cost grew with chain length: N=8 -> {calls_8} calls, "
        f"N=16 -> {calls_16} calls (shared cache must dedup identical junctions)"
    )
    # Small constant (end-effect junctions + one interior environment).
    assert calls_16 <= 5
    # Strictly sub-linear: the un-cached per-connection path would be N-1 (=15).
    assert calls_16 < 16 - 1


# --------------------------------------------------------------------------
# ac-001 — typed product == whole-chain baseline
# --------------------------------------------------------------------------
def test_product_types_match_whole_chain_baseline(opls_ff) -> None:
    opls = OplsTypifier(opls_ff, strict_typing=True)

    untyped = _build_chain(8, None)
    baseline = opls.typify(untyped.get_topo(gen_angle=True, gen_dihe=True))
    baseline_types = [a.get("type") for a in baseline.atoms]

    typed = _build_chain(8, opls)
    product_types = [a.get("type") for a in typed.atoms]

    assert all(t is not None for t in product_types)
    assert product_types == baseline_types
