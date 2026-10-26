"""Categories beyond the seven across the FFI seam: ``RelationStyle``.

The force-field IR is a protocol (``ff-ir-02-protocol`` §6): a category the
registry declares — molrec's ``drude``, or a custom ``urey_bradley`` of three
endpoints registered through ``mp.ff.ir`` — is a style category like ``bond``.
``ForceField.def_style`` returns a ``RelationStyle`` whose
``def_type(name, *endpoints, **params)`` takes as many endpoints as the
category's arity; its block ``<name>s`` is priced by its expression. A
category no registry declares survives a record and is refused, by name, when
something asks to price it without an expression.

The reference is LAMMPS ``angle_style charmm`` with K = 0: a pure 1-3
Urey–Bradley spring, ``k_ub (r13 − r_ub)²``.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import molpy as mp
import numpy as np
import pytest

UB_EXPRESSION = "k_ub*(distance(p1,p3)-r_ub)^2"
XYZ = np.array(
    [[0.0, 0.0, 0.0], [1.52, 0.1, 0.05], [2.1, 1.45, -0.1], [3.55, 1.6, 0.6]]
)
# (name, k_ub, r_ub), over atoms (0, 1, 2) and (1, 2, 3).
TYPES = [("t", 20.0, 2.45), ("u", 11.0, 2.2)]

# The public registration path: the process-wide registry gains the category.
mp.ff.style_registry.GlobalRegistry().register_category("urey_bradley", 3)


def _frame(block: str) -> mp.core.Frame:
    atoms = mp.core.Block()
    for d, key in enumerate("xyz"):
        atoms.insert(key, XYZ[:, d].copy())
    atoms.insert("type", ["A"] * 4)
    rows = mp.core.Block()
    for key, col in (("atomi", [0, 1]), ("atomj", [1, 2]), ("atomk", [2, 3])):
        rows.insert(key, np.array(col, dtype=np.uint32))
    rows.insert("type", [name for name, _, _ in TYPES])
    frame = mp.core.Frame()
    frame["atoms"] = atoms
    frame[block] = rows
    return frame


def _reference() -> tuple[float, np.ndarray]:
    ff = mp.ff.forcefield.ForceField("charmm", units="real")
    a = ff.def_style("atom", "full").def_type("A", mass=12.0)
    style = ff.def_style("angle", "charmm")
    for name, k_ub, r_ub in TYPES:
        style.def_type(name, a, a, a, k=0.0, theta0=109.5, k_ub=k_ub, r_ub=r_ub)
    frame = _frame("angles")
    return mp.ff.PotentialCompiler(ff).compile(frame).evaluate(frame)


def _ub_ff(category: str = "urey_bradley") -> mp.ff.forcefield.ForceField:
    ff = mp.ff.forcefield.ForceField("ub", units="real")
    a = ff.def_style("atom", "full").def_type("A", mass=12.0)
    style = ff.def_style(category, "spring", {"expression": UB_EXPRESSION})
    for name, k_ub, r_ub in TYPES:
        style.def_type(name, a, a, a, k_ub=k_ub, r_ub=r_ub)
    return ff


def _assert_prices_like_the_reference(ff: mp.ff.forcefield.ForceField) -> None:
    e_ref, f_ref = _reference()
    assert e_ref > 0.0
    frame = _frame("urey_bradleys")
    e, f = mp.ff.PotentialCompiler(ff).compile(frame).evaluate(frame)
    assert e == pytest.approx(e_ref, rel=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=0, atol=1e-12 * np.abs(f_ref).max())


def test_a_registered_category_is_a_relation_style() -> None:
    ff = _ub_ff()
    style = ff.get_style("urey_bradley", "spring")
    assert isinstance(style, mp.ff.forcefield.RelationStyle)
    assert (style.category, style.arity) == ("urey_bradley", 3)
    assert style.params == {"expression": UB_EXPRESSION}
    t = style.get_type_by_name("t")
    assert isinstance(t, mp.ff.forcefield.RelationType)
    assert (t.params["k_ub"], t.params["r_ub"]) == (20.0, 2.45)
    assert [e.name for e in t.endpoints] == ["A", "A", "A"]
    assert style in ff.get_styles(mp.ff.forcefield.Style)
    assert ff.get_styles("urey_bradley") == [style]
    assert ff.get_styles(mp.ff.forcefield.RelationStyle) == [style]
    assert len(ff.get_types(mp.ff.forcefield.RelationType)) == 2
    assert len(ff.get_types("urey_bradley")) == 2


def test_def_type_takes_exactly_the_arity_of_endpoints() -> None:
    ff = mp.ff.forcefield.ForceField("t", units="real")
    a = ff.def_style("atom", "full").def_type("A")
    style = ff.def_style("urey_bradley", "spring")
    with pytest.raises(ValueError, match="expected 3 endpoints, got 2"):
        style.def_type("x", a, a, k_ub=1.0)
    with pytest.raises(TypeError, match="AtomType"):
        style.def_type("x", a, a, "A")


def test_a_registered_category_prices_like_angle_charmm_without_k() -> None:
    _assert_prices_like_the_reference(_ub_ff())


def test_a_built_in_relation_category_needs_no_registration() -> None:
    ff = mp.ff.forcefield.ForceField("drude", units="real")
    atoms = ff.def_style("atom", "full")
    c, d = atoms.def_type("C"), atoms.def_type("D")
    style = ff.def_style("drude", "harmonic")
    assert isinstance(style, mp.ff.forcefield.RelationStyle)
    assert style.arity == 2
    style.def_type("C-D", c, d, k=500.0)
    assert ff.get_style("drude", "harmonic") == style


def test_an_unknown_category_is_refused() -> None:
    ff = mp.ff.forcefield.ForceField("t", units="real")
    with pytest.raises(ValueError, match="unknown"):
        ff.def_style("bespoke", "x")
    with pytest.raises(ValueError, match="unknown"):
        ff.get_styles("bespoke")


def test_a_relation_style_round_trips_through_a_section_a_store_and_a_pickle(
    tmp_path: Path,
) -> None:
    ff = _ub_ff()
    table = mp.io.mrec.ForceFieldSection.from_forcefield(ff).table(
        "urey_bradley", "spring"
    )
    assert list(table["k_ub"]) == [20.0, 11.0]
    path = tmp_path / "ff.mrec"
    mp.io.write_mrec(path, forcefield=ff)
    back = mp.io.mrec.MrecFile(path).forcefield().to_forcefield()
    style = back.get_style("urey_bradley", "spring")
    assert isinstance(style, mp.ff.forcefield.RelationStyle)
    assert style.params == {"expression": UB_EXPRESSION}
    _assert_prices_like_the_reference(back)
    _assert_prices_like_the_reference(pickle.loads(pickle.dumps(ff)))


def _unregistered_section(expression: str | None) -> mp.io.mrec.ForceFieldSection:
    """A section whose ``bespoke`` category nothing registers: the
    ``urey_bradley`` force field with its category renamed."""
    section = mp.io.mrec.ForceFieldSection.from_forcefield(_ub_ff())
    document = section.document
    tables = section.tables
    (entry,) = [s for s in document["styles"] if s["category"] == "urey_bradley"]
    entry["category"] = "bespoke"
    if expression is None:
        del entry["expression"]
    old = mp.io.mrec.ForceFieldSection.block_name("urey_bradley", "spring")
    tables[mp.io.mrec.ForceFieldSection.block_name("bespoke", "spring")] = tables.pop(
        old
    )
    return mp.io.mrec.ForceFieldSection(document, tables)


def test_an_unregistered_category_is_kept_and_refused_by_name_without_an_expression() -> (
    None
):
    back = _unregistered_section(None).to_forcefield()
    style = back.get_style("bespoke", "spring")
    assert isinstance(style, mp.ff.forcefield.RelationStyle)
    assert style.arity == 3
    assert [s.category for s in back.get_styles(mp.ff.forcefield.Style)] == [
        "atom",
        "bespoke",
    ]
    assert len(back.get_types("bespoke")) == 2
    assert (
        mp.io.mrec.ForceFieldSection.from_forcefield(back).table("bespoke", "spring")
        is not None
    )
    pickled = pickle.loads(pickle.dumps(back))
    assert pickled.get_style("bespoke", "spring").arity == 3
    compiler = mp.ff.PotentialCompiler(back)
    with pytest.raises(ValueError, match="no kernel for bespoke `spring`"):
        compiler.compile(_frame("bespokes"))
    # Without its block there is nothing to price.
    assert (
        compiler.compile(_frame("urey_bradleys")).evaluate(_frame("urey_bradleys"))[0]
        == 0.0
    )


def test_an_unregistered_category_with_an_expression_is_priced_by_it() -> None:
    back = _unregistered_section(UB_EXPRESSION).to_forcefield()
    e_ref, _ = _reference()
    frame = _frame("bespokes")
    e = mp.ff.PotentialCompiler(back).compile(frame).evaluate(frame)[0]
    assert e == pytest.approx(e_ref, rel=1e-12)
