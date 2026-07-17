"""``ForceFieldParams``: the second half of every force-field typifier.

It spends a node type rather than deciding one, so it is deliberately **not** a
:class:`~molpy.typifier.base.Typifier`. It is also the only place in molpy that
knows a ``Bond`` is parameterised by a ``BondType`` — arity cannot decide that, a
dihedral and an improper both span four atoms.

These tests carry over the behaviour of the three near-identical bonded typifiers
that used to live in ``typifier/atomistic.py``: a plain match, the reversed
orientation, an untyped endpoint, no match, and class-level matching.
"""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.core.atomistic import Angle, Bond, Dihedral
from molpy.core.forcefield import AngleType, AtomType, BondType, DihedralType
from molpy.typifier import ForceFieldParams, Typifier
from molpy.typifier._matching import TypeClassIndex
from molpy.typifier.forcefield import _FF_TYPE_OF, _TermMatcher


def _matcher(ff, ff_type, *, strict: bool = True) -> _TermMatcher:
    return _TermMatcher(ff, ff_type, TypeClassIndex(ff), strict=strict)


def _chain(*type_names: str) -> mp.Atomistic:
    """A straight chain of typed carbons, one atom per name."""
    s = mp.Atomistic()
    atoms = [
        s.def_atom(element="C", x=1.5 * i, y=0.0, z=0.0, type=name)
        for i, name in enumerate(type_names)
    ]
    for a, b in zip(atoms, atoms[1:], strict=False):
        s.def_bond(a, b)
    return s


class TestItIsNotATypifier:
    def test_it_decides_no_type_so_it_is_not_a_typifier(self):
        assert not issubclass(ForceFieldParams, Typifier)
        assert not hasattr(ForceFieldParams, "typify")

    def test_it_owns_the_only_link_to_forcefield_type_mapping(self):
        assert _FF_TYPE_OF[Bond] is BondType
        assert _FF_TYPE_OF[Angle] is AngleType
        assert _FF_TYPE_OF[Dihedral] is DihedralType

    def test_arity_could_not_have_decided_that_mapping(self):
        """A dihedral and an improper both span four atoms."""
        from molpy.core.atomistic import Improper
        from molpy.core.forcefield import ImproperType

        assert _FF_TYPE_OF[Improper] is ImproperType
        assert _FF_TYPE_OF[Dihedral] is not ImproperType


class TestBondMatching:
    @staticmethod
    def _ff():
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        ca = astyle.def_type("CA", type_="CA", class_="CT")
        ha = astyle.def_type("HA", type_="HA", class_="HC")
        bstyle = ff.def_bondstyle("harmonic")
        return ff, bstyle.def_type(ca, ha, k=1000.0, r0=1.08)

    def test_a_bond_gets_its_type_and_parameters(self):
        ff, bond_type = self._ff()
        bond = next(iter(_chain("CA", "HA").bonds))

        annotation = _matcher(ff, BondType).annotate(bond)

        assert annotation["type"] == bond_type.name
        assert annotation["k"] == 1000.0
        assert annotation["r0"] == 1.08

    def test_the_reversed_orientation_matches_identically(self):
        ff, bond_type = self._ff()
        forward = next(iter(_chain("CA", "HA").bonds))
        reverse = next(iter(_chain("HA", "CA").bonds))

        matcher = _matcher(ff, BondType)
        assert matcher.annotate(forward) == matcher.annotate(reverse)

    def test_an_untyped_endpoint_leaves_the_bond_undecided(self):
        """Undecided, not "typed as None": the caller keeps whatever it had."""
        ff, _ = self._ff()
        s = mp.Atomistic()
        a = s.def_atom(element="C", x=0.0, y=0.0, z=0.0, type="CA")
        b = s.def_atom(element="H", x=1.0, y=0.0, z=0.0)  # no type
        bond = s.def_bond(a, b)

        assert _matcher(ff, BondType).annotate(bond) == {}

    def test_no_matching_type_raises_when_strict(self):
        ff, _ = self._ff()
        bond = next(iter(_chain("CA", "CA").bonds))

        with pytest.raises(ValueError, match="No BondType found"):
            _matcher(ff, BondType).annotate(bond)

    def test_no_matching_type_is_undecided_when_not_strict(self):
        ff, _ = self._ff()
        bond = next(iter(_chain("CA", "CA").bonds))

        assert _matcher(ff, BondType, strict=False).annotate(bond) == {}

    def test_a_class_keyed_type_matches_an_atom_of_that_class(self):
        """This is what the old ``name.split("-")`` lookup in core could not do."""
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        # the bond type is keyed by the *class* placeholders, not the atom types
        ct = astyle.def_type("CT", type_="*", class_="CT")
        hc = astyle.def_type("HC", type_="*", class_="HC")
        astyle.def_type("CA", type_="CA", class_="CT")
        astyle.def_type("HA", type_="HA", class_="HC")
        bstyle = ff.def_bondstyle("harmonic")
        bond_type = bstyle.def_type(ct, hc, k=1.0, r0=1.0)

        bond = next(iter(_chain("CA", "HA").bonds))

        assert _matcher(ff, BondType).annotate(bond)["type"] == bond_type.name


class TestOverlayLayerTiebreak:
    """When two force-field types match a term equally well, the overlay wins.

    A tie needs two *different* component patterns scoring the same: here
    ``("A", "CY")`` scores exact+class = 3+1 and ``("CX", "B")`` scores
    class+exact = 1+3. Only then does the layer decide, and only then does
    declaration order stop deciding it by accident.

    Measured against the real CL&P force field, **no** bond assignment among 7140
    sampled atom-type pairs is settled this way — CL&P overrides OPLS-AA because
    its own types carry their own classes, which match different bonded types
    outright. So the tiebreak is a genuine but rarely-exercised rule, and
    ``test_clp.py`` does not cover it. It is pinned here instead.
    """

    @staticmethod
    def _tied_ff():
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        a = astyle.def_type("A", type_="A", class_="CX")  # base layer
        b = astyle.def_type("B", type_="B", class_="CY", layer=1)  # overlay
        cx = astyle.def_type("CX", type_="*", class_="CX")
        cy = astyle.def_type("CY", type_="*", class_="CY")

        bstyle = ff.def_bondstyle("harmonic")
        # the base-layer pattern is declared FIRST, so a matcher that ignores the
        # layer keeps it and never reaches the overlay
        base = bstyle.def_type(cx, b, k=2.0, r0=2.0)  # ("CX", "B") -> 1 + 3, layer 0
        overlay = bstyle.def_type(a, cy, k=1.0, r0=1.0)  # ("A", "CY") -> 3 + 1, layer 1
        return ff, base, overlay

    def test_the_two_patterns_really_do_tie_on_specificity(self):
        ff, _, _ = self._tied_ff()
        index = TypeClassIndex(ff)

        assert index.score(("CX", "B"), ["A", "B"]) == index.score(
            ("A", "CY"), ["A", "B"]
        )
        assert index.layer_of(("A", "CY")) > index.layer_of(("CX", "B"))

    def test_the_overlay_wins_the_tie(self):
        """Removing ``layer`` from the sort key (the ac-014 mutation) flips this."""
        ff, base, overlay = self._tied_ff()
        bond = next(iter(_chain("A", "B").bonds))

        annotation = _matcher(ff, BondType).annotate(bond)

        assert annotation["type"] == overlay.name
        assert annotation["k"] == 1.0
        assert annotation["type"] != base.name


class TestAngleAndDihedralMatching:
    def test_an_angle_matches_on_three_atom_types(self):
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        ha = astyle.def_type("HA", type_="HA", class_="HC")
        ca = astyle.def_type("CA", type_="CA", class_="CT")
        angle_type = ff.def_anglestyle("harmonic").def_type(
            ha, ca, ha, k=500.0, theta0=120.0
        )

        s = _chain("HA", "CA", "HA")
        angle = s.def_angle(*s.atoms)

        assert _matcher(ff, AngleType).annotate(angle)["type"] == angle_type.name

    def test_a_wildcard_dihedral_loses_to_a_fully_resolved_one(self):
        """``X-CT-CT-X`` must lose to the resolved pattern; OPLS relies on it."""
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        ct = astyle.def_type("CT", type_="CT", class_="CT")
        star = astyle.def_type("X", type_="*", class_="*")
        dstyle = ff.def_dihedralstyle("opls")
        dstyle.def_type(star, ct, ct, star, k1=1.0)
        resolved = dstyle.def_type(ct, ct, ct, ct, k1=9.0)

        s = _chain("CT", "CT", "CT", "CT")
        dihedral = s.def_dihedral(*s.atoms)

        annotation = _matcher(ff, DihedralType).annotate(dihedral)
        assert annotation["type"] == resolved.name
        assert annotation["k1"] == 9.0

    def test_an_untyped_endpoint_leaves_an_angle_undecided(self):
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        ca = astyle.def_type("CA", type_="CA", class_="CT")
        ff.def_anglestyle("harmonic").def_type(ca, ca, ca, k=1.0, theta0=1.0)

        s = mp.Atomistic()
        atoms = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(3)]
        atoms[0]["type"] = "CA"
        atoms[1]["type"] = "CA"  # atoms[2] left untyped
        angle = s.def_angle(*atoms)

        assert _matcher(ff, AngleType).annotate(angle) == {}


class TestForceFieldParams:
    @staticmethod
    def _ff():
        ff = mp.AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        ct = astyle.def_type("CT", type_="CT", class_="CT")
        ff.def_bondstyle("harmonic").def_type(ct, ct, k=1.0, r0=1.5)
        return ff

    def test_a_kind_the_forcefield_does_not_parameterize_is_left_alone(self):
        """No AngleType declared: the angles are the force field's silence, not a bug."""
        s = _chain("CT", "CT", "CT")
        s.def_angle(*s.atoms)

        typed = ForceFieldParams(self._ff()).assign(s)

        assert all(bond["type"] is not None for bond in typed.bonds)
        assert all(angle.get("type") is None for angle in typed.angles)

    def test_assign_parameterizes_a_graph_whose_types_are_already_known(self):
        typed = ForceFieldParams(self._ff()).assign(_chain("CT", "CT"))

        bond = next(iter(typed.bonds))
        assert bond["r0"] == 1.5

    def test_assign_does_not_mutate_its_input(self):
        source = _chain("CT", "CT")

        ForceFieldParams(self._ff()).assign(source)

        assert next(iter(source.bonds)).get("type") is None

    def test_an_unknown_link_kind_raises_rather_than_being_skipped(self):
        class _Exotic(Bond):
            pass

        params = ForceFieldParams(self._ff())
        params._terms.pop(Bond)  # simulate a kind absent from _FF_TYPE_OF

        with pytest.raises(TypeError, match="does not know which force-field type"):
            params.match(_chain("CT", "CT"))

    def test_the_forcefield_is_scanned_once_not_once_per_bonded_kind(self):
        ff = self._ff()
        real = ff.get_types
        calls = []

        def spy(cls):
            if cls is AtomType:
                calls.append(cls)
            return real(cls)

        ff.get_types = spy  # type: ignore[method-assign]
        ForceFieldParams(ff)

        assert len(calls) == 1
