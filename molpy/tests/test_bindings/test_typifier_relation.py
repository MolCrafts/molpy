"""A typifier emits terms of any relation kind (``ff-ir-02-protocol`` §6, WP6).

``TypeAssignment(nodes, links=...)`` keys its link rows by a relation class (``Bond``,
``Angle``, …) or by a block name (``"bonds"``, or a custom term block
``"urey_bradleys"`` the graph declares with ``declare_term`` and fills
through ``add_term``). A type
annotation under a kind defines a type of the category whose Frame block the
kind is, so a custom ``urey_bradley`` category is typified exactly like
``angle``: its terms land in the block ``urey_bradleys`` (``atomi`` …,
``type``) and price as LAMMPS ``angle_style charmm`` with K = 0.
"""

from __future__ import annotations

import molpy as mp
import numpy as np
import pytest
from molpy.core import Angle, Bond
from molpy.ff.typifier import TypeAssignment, Typifier

UB_EXPRESSION = "k_ub*(distance(p1,p3)-r_ub)^2"
XYZ = [(0.0, 0.0, 0.0), (1.52, 0.1, 0.05), (2.1, 1.45, -0.1), (3.55, 1.6, 0.6)]
ATOM_TYPES = ["A", "B", "C", "D"]
# (name, endpoints, k_ub, r_ub) over atoms (0, 1, 2) and (1, 2, 3).
TERMS = [("t", ("A", "B", "C"), 20.0, 2.45), ("u", ("B", "C", "D"), 11.0, 2.2)]

# The public registration path: the process-wide registry gains the category.
mp.ff.style_registry.register_category("urey_bradley", 3)


def _chain(kind: str) -> mp.core.Atomistic:
    """Four atoms with two three-atom relations of ``kind``."""
    mol = mp.core.Atomistic()
    atoms = [mol.add_atom(atomic_number=6, x=x, y=y, z=z) for x, y, z in XYZ]
    if kind != "angles":
        mol.declare_term(kind, 3)
    for i in range(2):
        if kind == "angles":
            mol.add_angle(*atoms[i : i + 3])
        else:
            mol.add_term(kind, atoms[i : i + 3])
    return mol


class _Chain(Typifier):
    """Types atoms ``A`` … ``D`` and the two terms of ``key`` under ``style``."""

    def __init__(self, key: object, category: str, style: str, **extra: float) -> None:
        self.key, self.category, self.style, self.extra = key, category, style, extra
        self.style_params = (
            {"expression": UB_EXPRESSION} if category == "urey_bradley" else {}
        )

    def assign(self, graph: mp.core.Atomistic) -> TypeAssignment:
        nodes = [{"type": ("full", t, (), {"mass": 12.0})} for t in ATOM_TYPES]
        rows = [
            {
                "type": (
                    self.style,
                    name,
                    ends,
                    {"k_ub": k_ub, "r_ub": r_ub, **self.extra},
                )
            }
            for name, ends, k_ub, r_ub in TERMS
        ]
        return TypeAssignment(
            nodes,
            {self.key: rows},
            styles=[
                ("atom", "full", {}),
                (self.category, self.style, self.style_params),
            ],
        )


def _energy_forces(
    typifier: Typifier, kind: str
) -> tuple[float, np.ndarray, mp.core.Frame]:
    frame = typifier.typify(_chain(kind)).to_frame()
    ff = typifier.forcefield()
    e, f = mp.ff.compile.PotentialCompiler(ff).compile(frame).evaluate(frame)
    return e, f, frame


def test_a_custom_relation_kind_is_typified_and_priced_like_angle_charmm() -> None:
    ub = _Chain("urey_bradleys", "urey_bradley", "spring")
    e, f, frame = _energy_forces(ub, "urey_bradleys")

    block = frame["urey_bradleys"]
    assert block["atomi"].tolist() == [0, 1]
    assert block["atomj"].tolist() == [1, 2]
    assert block["atomk"].tolist() == [2, 3]
    assert block["type"].tolist() == ["t", "u"]
    style = ub.forcefield().get_style("urey_bradley", "spring")
    assert isinstance(style, mp.ff.forcefield.RelationStyle)
    assert [t.name for t in style.get_types()] == ["t", "u"]
    assert [e.name for e in style.get_type_by_name("t").endpoints] == ["A", "B", "C"]

    charmm = _Chain(Angle, "angle", "charmm", k=0.0, theta0=109.5)
    e_ref, f_ref, _ = _energy_forces(charmm, "angles")
    assert e_ref > 0.0
    assert e == pytest.approx(e_ref, rel=1e-12)
    np.testing.assert_allclose(f, f_ref, rtol=0, atol=1e-12 * np.abs(f_ref).max())


def test_a_built_in_kind_by_name_is_the_kind_by_class() -> None:
    by_class = _Chain(Angle, "angle", "charmm", k=0.0, theta0=109.5)
    by_name = _Chain("angles", "angle", "charmm", k=0.0, theta0=109.5)
    a = by_class.typify(_chain("angles")).to_frame()["angles"]["type"].tolist()
    b = by_name.typify(_chain("angles")).to_frame()["angles"]["type"].tolist()
    assert a == b == ["t", "u"]


def test_a_kind_named_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="'bonds' more than once"):
        TypeAssignment([], {Bond: [], "bonds": []})


@pytest.mark.parametrize("key", [3, mp.core.Atom])
def test_a_key_that_names_no_kind_is_refused(key: object) -> None:
    with pytest.raises(TypeError, match="relation kind names"):
        TypeAssignment([], {key: []})


def test_a_kind_the_graph_lacks_is_refused_by_name() -> None:
    typifier = _Chain("cross_terms", "urey_bradley", "spring")
    with pytest.raises(ValueError, match="graph has no 'cross_terms' relation kind"):
        typifier.typify(_chain("angles"))
    assert typifier.forcefield().get_styles(mp.ff.forcefield.Style) == []


def test_repr_lists_the_kinds() -> None:
    m = TypeAssignment([{}], {Bond: [{}], "urey_bradleys": [{}, {}]})
    assert (
        repr(m)
        == "TypeAssignment(nodes=1, links={bonds=1, urey_bradleys=2}, styles=0, pairs=0)"
    )
