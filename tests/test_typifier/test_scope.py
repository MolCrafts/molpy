"""TypeScope / SmartsTypifier — pattern-derived receptive field (graph-assembler-04)."""

from __future__ import annotations

import re
from pathlib import Path

import molrs
import pytest

from molpy.typifier import SmartsTypifier, TypeScope, UnboundedPatternSet
from molpy.typifier.scope import TypeScope as TypeScopeDirect


def test_molrs_syntax_facts_ready():
    assert hasattr(molrs.SmartsPattern, "max_bond_depth")
    assert hasattr(molrs.SmartsPattern, "ring_primitives")
    assert hasattr(molrs.Atomistic, "max_ring_system_size")
    pin = Path("pyproject.toml").read_text(encoding="utf-8")
    assert "molcrafts-molrs==" in pin


def test_type_scope_from_linear_patterns():
    scope = TypeScope.from_patterns(["C", "CC", "CCO"])
    # max depth 2, TERM_REACH 2 → 2
    assert scope.reach == 2
    assert scope.interior_reach == 2
    assert scope.extract_radius == 4


def test_type_scope_sized_ring_contributes():
    # [r6] → ⌊6/2⌋+1 = 4
    scope = TypeScope.from_patterns(["[r6]"])
    assert scope.reach == 4


@pytest.mark.parametrize(
    "smarts,primitive_substr",
    [
        ("[R]", "membership"),
        ("[R0]", "membership"),
        ("[R2]", "ring_count"),
        ("[x2]", "ring_bond_count"),
    ],
)
def test_unbounded_primitives_raise(smarts, primitive_substr):
    with pytest.raises(UnboundedPatternSet) as ei:
        TypeScope.from_patterns([smarts])
    assert primitive_substr in ei.value.primitive


def test_smarts_typifier_unbounded_is_typeerror_naming_primitive():
    with pytest.raises(TypeError, match="membership|ring_count|ring_bond_count") as ei:
        SmartsTypifier(["[R]"])
    assert (
        "region typing" in str(ei.value).lower() or "receptive" in str(ei.value).lower()
    )


def test_smarts_typifier_sized_ok():
    typ = SmartsTypifier(["[r6]", "CCO"])
    assert typ.scope.reach >= 2
    assert typ.scope.reach == max(2, 2, 6 // 2 + 1)  # 4


def test_replacing_R_with_r6_makes_set_bounded():
    with pytest.raises(TypeError):
        SmartsTypifier(["[R]C"])
    typ = SmartsTypifier(["[r6]C"])
    assert typ.scope.reach >= 1


def test_naphthalene_ring_system_size_is_ten():
    g = molrs.Atomistic()
    ids = [g.add_atom("C", float(i), 0.0, 0.0) for i in range(10)]
    # Ring A 0-5
    for i in range(5):
        g.add_bond(ids[i], ids[i + 1])
    g.add_bond(ids[5], ids[0])
    # Ring B fused at 2-3
    g.add_bond(ids[3], ids[6])
    g.add_bond(ids[6], ids[7])
    g.add_bond(ids[7], ids[8])
    g.add_bond(ids[8], ids[9])
    g.add_bond(ids[9], ids[2])
    assert g.max_ring_system_size() == 10


def test_benzene_ring_system_size_is_six():
    g = molrs.Atomistic()
    ids = [g.add_atom("C", float(i), 0.0, 0.0) for i in range(6)]
    for i in range(6):
        g.add_bond(ids[i], ids[(i + 1) % 6])
    assert g.max_ring_system_size() == 6


def test_term_reach_is_only_literal_in_typifier_scope_sources():
    """ac-006: no reach=<digit> literals under typifier except TERM_REACH."""
    root = Path("src/molpy/typifier")
    hits: list[str] = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r"reach\s*=\s*[0-9]", line):
                hits.append(f"{path}:{i}:{line.strip()}")
    # Only TypeScope.TERM_REACH = 2 is allowed as a documented non-magic constant.
    assert hits == [], f"unexpected reach literals: {hits}"
    assert TypeScopeDirect.TERM_REACH == 2


def test_empty_patterns_rejected():
    with pytest.raises(ValueError):
        TypeScope.from_patterns([])
    with pytest.raises(ValueError):
        SmartsTypifier([])


def test_bounded_pattern_set_exports_measured_opls_floor():
    """ac-004 (bounded path): max depth 2 + no untyped rings → reach == measured min.

    Full OPLS XML defs include ``[R2]`` (unbounded); production OPLS stays on
    ``OPLSAATypifier`` + assembler ``reach=``. A *bounded* pattern set whose
    max_bond_depth is 2 must export ``reach == 2`` — the measured PEO floor
    from graph-assembler-01 ac-003.
    """
    # Patterns with diameter ≤ 2 and only sized / no ring tokens.
    patterns = ["[C]", "CC", "CCO", "[CX4](C)(C)(H)H"]
    scope = TypeScope.from_patterns(patterns)
    assert scope.reach == 2
    typ = SmartsTypifier(patterns)
    assert typ.scope.reach == 2


def test_mmff_like_bounded_ring_exports_sized_contrib():
    """Sized aromatic six-ring contributes ⌊6/2⌋+1 = 4, not unbounded membership."""
    scope = TypeScope.from_patterns(["[c]1[c][c][c][c][c]1", "[r6]"])
    assert scope.reach == max(
        2, molrs.SmartsPattern("[c]1[c][c][c][c][c]1").max_bond_depth, 4
    )
