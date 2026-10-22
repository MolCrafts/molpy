"""Seam tests for the ``molrs.ff.forcefield.ForceField`` construction primitives.

A force field is built through two doors: ``ForceField.def_style(category,
name, params=None)`` and the per-category ``def_type`` of the returned style
handle — the type's name, then its endpoints (``AtomType`` handles), then
keyword params — which returns the type's handle. The Rust builder semantics
(style identity, arity, the conflict rule, names never split) are unit-tested
in ``crates/molrs/src/ff/forcefield/mod.rs``. These tests prove only the binding
seam: the typed doors, the str/float split of params, and error mapping.
"""

import pickle

import molrs
import numpy as np
import pytest
from molrs import _native

HANDLE_CLASSES = (
    "ForceField",
    "Style",
    "AtomStyle",
    "BondStyle",
    "AngleStyle",
    "DihedralStyle",
    "ImproperStyle",
    "PairStyle",
    "ForceFieldType",
    "AtomType",
    "BondType",
    "AngleType",
    "DihedralType",
    "ImproperType",
    "PairType",
)


def _rows(style) -> list[tuple[str, dict]]:
    """``(name, params)`` of every type of ``style``, in definition order."""
    return [(t.name, t.params) for t in style.get_types()]


@pytest.mark.parametrize("name", HANDLE_CLASSES)
def test_the_force_field_classes_are_the_native_classes(name):
    assert getattr(molrs.ff.forcefield, name) is getattr(_native, name)


def test_the_force_field_class_can_be_subclassed():
    """molnex's ForceField extends this one; a subclass keeps the base's state."""

    class Sub(molrs.ff.forcefield.ForceField):
        pass

    ff = Sub("scratch")
    assert isinstance(ff, molrs.ff.forcefield.ForceField)
    assert ff.name == "scratch"


class _Extended(molrs.ff.forcefield.ForceField):
    """Module level, so pickle can find it."""


def test_a_force_field_subclass_pickles_as_itself():
    ff = _Extended("ext", units="metal")
    (ct,) = _atoms(ff, "CT")
    ff.def_style("pair", "lj/cut").def_type("CT", ct, epsilon=0.1, sigma=3.5)
    ff.origin = "molnex"
    back = pickle.loads(pickle.dumps(ff))
    assert type(back) is _Extended
    assert back.origin == "molnex"
    assert (back.name, back.units) == ("ext", "metal")
    assert _rows(back.get_style("pair", "lj/cut")) == [
        ("CT", {"epsilon": 0.1, "sigma": 3.5})
    ]


def test_special_bonds_reads_the_declared_triples():
    ff = molrs.ff.forcefield.ForceField("sb")
    assert ff.special_bonds == ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])
    assert ff.special_bonds == ([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])


def test_empty_forcefield_constructs():
    ff = molrs.ff.forcefield.ForceField("scratch")
    assert ff.name == "scratch"
    assert ff.get_styles(molrs.ff.forcefield.Style) == []


def _atoms(
    ff: molrs.ff.forcefield.ForceField, *names: str
) -> list[molrs.ff.forcefield.AtomType]:
    atom_style = ff.def_style("atom", "full")
    return [atom_style.def_type(name, mass=1.0) for name in names]


def test_atom_def_type_returns_the_atom_type_handle():
    ff = molrs.ff.forcefield.ForceField("atoms")
    ct = ff.def_style("atom", "full").def_type("CT", mass=12.011, element="C")
    assert isinstance(ct, molrs.ff.forcefield.AtomType)
    assert ct.name == "CT"
    assert _rows(ff.get_style("atom", "full")) == [
        ("CT", {"mass": 12.011, "element": "C"})
    ]


def test_bond_def_type_stores_name_and_endpoints_as_given():
    ff = molrs.ff.forcefield.ForceField("bonds")
    ct, oh = _atoms(ff, "CT", "OH")
    bond = ff.def_style("bond", "harmonic").def_type(
        "anything", ct, oh, k=300.0, r0=1.4
    )
    assert isinstance(bond, molrs.ff.forcefield.BondType)
    assert bond.name == "anything"
    assert (bond.itom.name, bond.jtom.name) == ("CT", "OH")
    assert _rows(ff.get_style("bond", "harmonic")) == [
        ("anything", {"k": 300.0, "r0": 1.4})
    ]


def test_a_dashed_name_is_never_split():
    ff = molrs.ff.forcefield.ForceField("names")
    hc, os_ = _atoms(ff, "HC", "OS")
    bond = ff.def_style("bond", "harmonic").def_type("CT-OH", hc, os_, k=1.0, r0=1.0)
    assert bond.endpoints == (hc, os_)


def test_angle_dihedral_improper_def_type_return_typed_handles():
    ff = molrs.ff.forcefield.ForceField("bonded")
    hc, ct, oh = _atoms(ff, "HC", "CT", "OH")
    angle = ff.def_style("angle", "harmonic").def_type(
        "HC-CT-OH", hc, ct, oh, k=70.0, theta0=108.9
    )
    dihedral = ff.def_style("dihedral", "opls").def_type(
        "HC-CT-CT-OH", hc, ct, ct, oh, k1=0.0, k2=0.0, k3=0.3, k4=0.0
    )
    improper = ff.def_style("improper", "harmonic").def_type(
        "HC-OH-CT-HC", hc, oh, ct, hc, k=2.0, chi0=0.0
    )
    assert isinstance(angle, molrs.ff.forcefield.AngleType)
    assert [e.name for e in angle.endpoints] == ["HC", "CT", "OH"]
    assert isinstance(dihedral, molrs.ff.forcefield.DihedralType)
    assert [e.name for e in dihedral.endpoints] == ["HC", "CT", "CT", "OH"]
    assert isinstance(improper, molrs.ff.forcefield.ImproperType)
    assert [e.name for e in improper.endpoints] == ["HC", "OH", "CT", "HC"]
    # Every slot `def_type` names has its accessor.
    assert (angle.itom, angle.jtom, angle.ktom) == (hc, ct, oh)
    assert (dihedral.ktom, dihedral.ltom) == (ct, oh)
    assert (improper.ktom, improper.ltom) == (ct, hc)


def test_pair_def_type_without_jtom_is_the_self_pair():
    ff = molrs.ff.forcefield.ForceField("pairs")
    ct, oh = _atoms(ff, "CT", "OH")
    pair_style = ff.def_style("pair", "lj/cut")
    self_pair = pair_style.def_type("CT", ct, epsilon=0.066, sigma=3.5)
    cross = pair_style.def_type("CT-OH", ct, oh, epsilon=0.1, sigma=3.3)
    assert isinstance(self_pair, molrs.ff.forcefield.PairType)
    assert (self_pair.itom.name, self_pair.jtom.name) == ("CT", "CT")
    assert (cross.itom.name, cross.jtom.name) == ("CT", "OH")


def test_an_endpoint_that_is_not_an_atom_type_raises_type_error():
    ff = molrs.ff.forcefield.ForceField("guard")
    (ct,) = _atoms(ff, "CT")
    with pytest.raises(TypeError, match="AtomType"):
        ff.def_style("bond", "harmonic").def_type("CT-OH", ct, "OH", k=1.0)


def test_an_existing_types_endpoints_are_handles_def_type_accepts():
    ff = molrs.ff.forcefield.ForceField("reuse")
    ct, oh = _atoms(ff, "CT", "OH")
    bond = ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=1.0, r0=1.0)
    again = ff.def_style("bond", "morse").def_type("CT-OH", *bond.endpoints, d0=1.0)
    assert [e.name for e in again.endpoints] == ["CT", "OH"]


def test_def_style_returns_the_category_handle():
    ff = molrs.ff.forcefield.ForceField("handle")
    assert isinstance(ff.def_style("pair", "lj/cut"), molrs.ff.forcefield.PairStyle)


def test_style_params_exposes_mixing():
    ff = molrs.ff.forcefield.ForceField("lj")
    style = ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "geometric"})
    assert style.params == {"cutoff": 10.0, "mixing": "geometric"}


def test_kspace_is_not_a_category():
    ff = molrs.ff.forcefield.ForceField("guard")
    with pytest.raises(ValueError, match="unknown"):
        ff.def_style("kspace", "pme")


def test_a_missing_style_is_none():
    ff = molrs.ff.forcefield.ForceField("empty")
    assert ff.get_style("bond", "nope") is None


def test_conflicting_def_type_raises_value_error():
    ff = molrs.ff.forcefield.ForceField("conflict")
    ct, oh = _atoms(ff, "CT", "OH")
    style = ff.def_style("bond", "harmonic")
    style.def_type("CT-OH", ct, oh, k=300.0, r0=1.4)
    with pytest.raises(ValueError):
        style.def_type("CT-OH", ct, oh, k=310.0, r0=1.4)


def test_conflicting_def_style_raises_value_error():
    ff = molrs.ff.forcefield.ForceField("conflict")
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
    with pytest.raises(ValueError):
        ff.def_style("pair", "lj/cut", {"cutoff": 12.0})


# ---- merge: one Rust implementation behind the Python seam ----


def test_merge_returns_self():
    ff = molrs.ff.forcefield.ForceField("target")
    other = molrs.ff.forcefield.ForceField("source")
    ct, oh = _atoms(other, "CT", "OH")
    other.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=300.0, r0=1.4)
    assert ff.merge(other) is ff
    assert _rows(ff.get_style("bond", "harmonic")) == [
        ("CT-OH", {"k": 300.0, "r0": 1.4})
    ]


def test_conflicting_merge_raises_value_error():
    ff = molrs.ff.forcefield.ForceField("target")
    ct, oh = _atoms(ff, "CT", "OH")
    ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=300.0, r0=1.4)
    other = molrs.ff.forcefield.ForceField("source")
    ct, oh = _atoms(other, "CT", "OH")
    other.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=310.0, r0=1.4)
    with pytest.raises(ValueError):
        ff.merge(other)


# ---- units ----


def test_units_default_to_real():
    assert molrs.ff.forcefield.ForceField("plain").units == "real"


def test_units_ctor_arg_declares_units():
    assert molrs.ff.forcefield.ForceField("reduced", units="lj").units == "lj"


# ---- readers return the one class, with full style params ----

_OPLS_GEOMETRIC = """<ForceField name="OPLS-AA" combining_rule="geometric">
  <AtomTypes>
    <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
  </AtomTypes>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_001" charge="0.0" sigma="0.375" epsilon="0.43932"/>
  </NonbondedForce>
</ForceField>"""


def test_read_opls_xml_returns_the_force_field_with_its_mixing(tmp_path):
    path = tmp_path / "opls.xml"
    path.write_text(_OPLS_GEOMETRIC, encoding="utf-8")
    ff = molrs.io.openmm_xml.OpenmmXmlReader().read(path)
    assert type(ff) is molrs.ff.forcefield.ForceField
    assert ff.get_style("pair", "lj/cut")["mixing"] == "geometric"


def test_a_written_force_field_reads_back_through_a_pathlike(tmp_path):
    ff = molrs.ff.forcefield.ForceField("round")
    (ct,) = _atoms(ff, "CT")
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=300.0, r0=1.5)
    path = tmp_path / "ff.xml"
    molrs.io.openmm_xml.OpenmmXmlWriter().write(path, ff)
    back = molrs.io.openmm_xml.OpenmmXmlReader().read(path)
    assert type(back) is molrs.ff.forcefield.ForceField
    # The file is in nm and kJ/mol: the trip is exact to the conversions' ulp.
    ((name, params),) = _rows(back.get_style("bond", "harmonic"))
    assert name == "CT-CT"
    assert params == pytest.approx({"k": 300.0, "r0": 1.5}, rel=1e-14)


# ---- handles: queries, params, equality ----


def test_get_styles_and_get_types_select_by_category_or_class():
    ff = molrs.ff.forcefield.ForceField("select")
    ct, oh = _atoms(ff, "CT", "OH")
    ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=1.0, r0=1.0)
    ff.def_style("bond", "morse").def_type("CT-OH", ct, oh, d0=1.0)
    assert [s.name for s in ff.get_styles("bond")] == ["harmonic", "morse"]
    assert [s.name for s in ff.get_styles(molrs.ff.forcefield.BondStyle)] == [
        "harmonic",
        "morse",
    ]
    assert len(ff.get_styles(molrs.ff.forcefield.Style)) == 3
    assert [t.name for t in ff.get_types("atom")] == ["CT", "OH"]
    assert len(ff.get_types(molrs.ff.forcefield.BondType)) == 2
    assert len(ff.get_types(molrs.ff.forcefield.ForceFieldType)) == 4


def test_type_equality_includes_the_style():
    ff = molrs.ff.forcefield.ForceField("styles")
    ct, oh = _atoms(ff, "CT", "OH")
    harmonic = ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=1.0, r0=1.0)
    morse = ff.def_style("bond", "morse").def_type("CT-OH", ct, oh, d0=1.0)
    assert harmonic != morse
    assert len({harmonic, morse}) == 2
    assert harmonic == ff.get_style("bond", "harmonic").get_type_by_name("CT-OH")


def test_handle_equality_includes_the_force_field():
    a, b = molrs.ff.forcefield.ForceField("a"), molrs.ff.forcefield.ForceField("b")
    (ct_a,) = _atoms(a, "CT")
    (ct_b,) = _atoms(b, "CT")
    style_a = a.def_style("bond", "harmonic")
    style_b = b.def_style("bond", "harmonic")
    assert style_a != style_b and len({style_a, style_b}) == 2
    assert ct_a != ct_b and len({ct_a, ct_b}) == 2
    assert style_a == a.get_style("bond", "harmonic")


def test_a_type_param_can_be_a_string():
    ff = molrs.ff.forcefield.ForceField("strings")
    (ct,) = _atoms(ff, "CT")
    ct["element"] = "C"
    ct["charge"] = -0.5
    assert ct.params == {"mass": 1.0, "element": "C", "charge": -0.5}
    with pytest.raises(TypeError, match="a number, a str or an array"):
        ct["tags"] = {"a": 1}
    with pytest.raises(TypeError, match="rectangular"):
        ct["tags"] = ["a", "b"]


def test_endpoints_are_the_defined_atom_types():
    ff = molrs.ff.forcefield.ForceField("ends")
    ct, oh = _atoms(ff, "CT", "OH")
    bond = ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=1.0, r0=1.0)
    assert bond.endpoints == (ct, oh)
    assert bond.itom.params == {"mass": 1.0}


# ---- pickle ----


def test_a_force_field_pickles_with_its_whole_definition():
    ff = molrs.ff.forcefield.ForceField("full", units="metal")
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.75])
    ct, oh = _atoms(ff, "CT", "OH")
    ct["element"] = "C"
    ff.def_style("bond", "harmonic").def_type("CT-OH", ct, oh, k=300.0, r0=1.4)
    ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "geometric"}).def_type(
        "CT", ct, epsilon=0.1, sigma=3.5
    )
    back = pickle.loads(pickle.dumps(ff))
    assert type(back) is molrs.ff.forcefield.ForceField
    assert (back.name, back.units) == ("full", "metal")
    assert [
        (s.category, s.name, s.params)
        for s in back.get_styles(molrs.ff.forcefield.Style)
    ] == [
        (s.category, s.name, s.params) for s in ff.get_styles(molrs.ff.forcefield.Style)
    ]
    assert [
        (t.name, t.params, [e.name for e in t.endpoints])
        for t in back.get_types(molrs.ff.forcefield.ForceFieldType)
    ] == [
        (t.name, t.params, [e.name for e in t.endpoints])
        for t in ff.get_types(molrs.ff.forcefield.ForceFieldType)
    ]
    # The special bonds survive: merging the original into the copy agrees.
    assert back.merge(ff) is back


# ---- PotentialCompiler: the one door from a ForceField to kernels ----
#
# The compile semantics (skipping absent blocks, the 1.5 kcal/mol closed form,
# refusals) are unit-tested in ``crates/molrs/src/ff/compile/mod.rs``. These
# tests prove only the binding seam: construction, the three doors, their
# return types, error mapping and the single public path.


def _bond_ff() -> molrs.ff.forcefield.ForceField:
    ff = molrs.ff.forcefield.ForceField("compile")
    (ct,) = _atoms(ff, "CT")
    ff.def_style("bond", "harmonic").def_type("CT-CT", ct, ct, k=300.0, r0=1.5)
    return ff


def _bonded_pair(label: str = "CT-CT") -> molrs.core.Frame:
    atoms = molrs.core.Block()
    for key, values in (("x", [0.0, 1.6]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])):
        atoms.insert(key, np.array(values))
    bonds = molrs.core.Block()
    bonds.insert("atomi", np.array([0], dtype=np.uint32))
    bonds.insert("atomj", np.array([1], dtype=np.uint32))
    bonds.insert("type", [label])
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["bonds"] = bonds
    return frame


def test_potential_compiler_constructs_from_a_forcefield():
    compiler = molrs.ff.compile.PotentialCompiler(_bond_ff())
    assert isinstance(compiler, molrs.ff.compile.PotentialCompiler)


def test_compile_returns_potentials():
    pots = molrs.ff.compile.PotentialCompiler(_bond_ff()).compile(_bonded_pair())
    assert isinstance(pots, molrs.ff.potential.Potentials)
    assert len(pots) == 1


def test_compile_none_raises_type_error():
    compiler = molrs.ff.compile.PotentialCompiler(_bond_ff())
    with pytest.raises(TypeError):
        compiler.compile(None)  # type: ignore[arg-type]


def test_defer_returns_empty_potentials_that_bind_on_evaluation():
    compiler = molrs.ff.compile.PotentialCompiler(_bond_ff())
    deferred = compiler.defer()
    assert isinstance(deferred, molrs.ff.potential.Potentials)
    assert len(deferred) == 0
    frame = _bonded_pair()
    assert deferred.evaluate(frame)[0] == compiler.compile(frame).evaluate(frame)[0]


def test_compile_typed_returns_typed_potentials():
    from molrs._native import WeightedTerms

    typed = molrs.ff.compile.PotentialCompiler(_bond_ff()).compile_typed(_bonded_pair())
    assert isinstance(typed, WeightedTerms)
    assert len(typed) == 1


def test_compile_unknown_type_label_raises_value_error():
    compiler = molrs.ff.compile.PotentialCompiler(_bond_ff())
    with pytest.raises(ValueError):
        compiler.compile(_bonded_pair("XX-XX"))


def _lj_ab(cross: bool) -> molrs.ff.forcefield.ForceField:
    """Self rows A (0.1, 3.0) and B (0.4, 3.6), geometric mixing, and with
    ``cross`` an explicit A-B row (0.9, 2.0)."""
    ff = molrs.ff.forcefield.ForceField("nbfix")
    a, b = _atoms(ff, "A", "B")
    lj = ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "geometric"})
    lj.def_type("A", a, epsilon=0.1, sigma=3.0)
    lj.def_type("B", b, epsilon=0.4, sigma=3.6)
    if cross:
        lj.def_type("A-B", a, b, epsilon=0.9, sigma=2.0)
    return ff


def _lj_pair_energy(ff: molrs.ff.forcefield.ForceField, r: float) -> float:
    atoms = molrs.core.Block()
    for key, values in (("x", [0.0, r]), ("y", [0.0, 0.0]), ("z", [0.0, 0.0])):
        atoms.insert(key, np.array(values))
    atoms.insert("type", ["A", "B"])
    pairs = molrs.core.Block()
    pairs.insert("atomi", np.array([0], dtype=np.uint64))
    pairs.insert("atomj", np.array([1], dtype=np.uint64))
    frame = molrs.core.Frame()
    frame["atoms"] = atoms
    frame["pairs"] = pairs
    return molrs.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)[0]


def _lj(eps: float, sigma: float, r: float) -> float:
    s6 = (sigma / r) ** 6
    return 4.0 * eps * (s6 * s6 - s6)


def test_an_explicit_cross_row_overrides_the_mixing_rule():
    r = 2.5
    mixed = _lj((0.1 * 0.4) ** 0.5, (3.0 * 3.6) ** 0.5, r)
    assert _lj_pair_energy(_lj_ab(cross=False), r) == pytest.approx(mixed, rel=1e-12)
    assert _lj_pair_energy(_lj_ab(cross=True), r) == pytest.approx(
        _lj(0.9, 2.0, r), rel=1e-12
    )


def test_a_pair_restated_in_reverse_is_one_row_or_a_conflict():
    """A pair is found by its unordered endpoints, so a style holds one row
    per pair: an equal restatement under another name is a no-op, a
    different one is refused and the first row stands."""
    ff = _lj_ab(cross=True)
    lj = ff.get_style("pair", "lj/cut")
    a, b = _atoms(ff, "A", "B")
    restated = lj.def_type("B-A", b, a, epsilon=0.9, sigma=2.0, desc="restated")
    assert restated.name == "A-B"
    assert len(ff.get_style("pair", "lj/cut").get_types()) == 3
    with pytest.raises(ValueError, match="restates the pair"):
        lj.def_type("nbfix", a, b, epsilon=0.8, sigma=2.0)
    assert _lj_pair_energy(ff, 2.5) == pytest.approx(_lj(0.9, 2.0, 2.5), rel=1e-12)


def test_potential_compiler_lives_in_compile():
    assert hasattr(molrs.ff.compile, "PotentialCompiler")
    assert not hasattr(molrs.ff, "PotentialCompiler")


def test_style_setitem_declares_a_cutoff():
    ff = molrs.ff.forcefield.ForceField("t")
    style = ff.def_style("pair", "lj/cut")
    style["cutoff"] = 12.0
    assert ff.get_style("pair", "lj/cut")["cutoff"] == 12.0


def test_style_setitem_takes_a_string_param():
    ff = molrs.ff.forcefield.ForceField("t")
    style = ff.def_style("pair", "lj/cut")
    style["mixing"] = "geometric"
    assert ff.get_style("pair", "lj/cut")["mixing"] == "geometric"
    with pytest.raises(TypeError, match="numbers of one rectangular shape"):
        style["mixing"] = ["geometric"]
    with pytest.raises(TypeError, match="a number, a str or an array of numbers"):
        style["mixing"] = object()


def test_style_setitem_takes_every_param_value_kind():
    """``Style::set_param`` takes any ``ParamValue``: a number, a str, or an
    array of numbers (stored as float64), as a type's ``__setitem__`` does."""
    ff = molrs.ff.forcefield.ForceField("t")
    style = ff.def_style("pair", "lj/cut")
    style["cutoff"] = 10.0
    style["table"] = [[1, 2], [3, 4]]
    got = ff.get_style("pair", "lj/cut")
    assert got["cutoff"] == 10.0
    np.testing.assert_array_equal(got["table"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert got["table"].dtype == np.float64


def test_style_del_type_removes_one_type_by_name():
    ff = molrs.ff.forcefield.ForceField("t")
    atoms = ff.def_style("atom", "full")
    c, h = atoms.def_type("C", mass=12.0), atoms.def_type("H", mass=1.0)
    bonds = ff.def_style("bond", "harmonic")
    bonds.def_type("C-H", c, h, k=300.0, r0=1.1)
    bonds.def_type("C-C", c, c, k=300.0, r0=1.5)
    assert bonds.del_type("C-H") == 1
    assert [t.name for t in bonds.get_types()] == ["C-C"]
    assert bonds.get_type_by_name("C-H") is None
    assert bonds.del_type("C-H") == 0
    assert [t.name for t in ff.get_types("bond")] == ["C-C"]
