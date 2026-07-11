"""The specificity contest that decides which bonded force-field type wins.

Carries over the rules the old module-level ``atomtype_matches`` encoded — exact
type beats class beats wildcard, and a component that matches neither kills the
whole pattern — except that they are now *scored* rather than merely accepted or
rejected, so the most specific type can win.
"""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.typifier._matching import TypeClassIndex


def _forcefield() -> mp.AtomisticForcefield:
    """opls_135 (class CT), opls_140 (class HC), and a layer-1 CL&P overlay type."""
    ff = mp.AtomisticForcefield()
    astyle = ff.def_atomstyle("full")
    astyle.def_type("opls_135", type_="opls_135", class_="CT")
    astyle.def_type("opls_140", type_="opls_140", class_="HC")
    astyle.def_type("clp_C1", type_="clp_C1", class_="CR", layer=1)
    return ff


class TestTypeClassIndex:
    def test_class_of_resolves_a_type_to_its_class(self):
        index = TypeClassIndex(_forcefield())

        assert index.class_of("opls_135") == "CT"
        assert index.class_of("opls_140") == "HC"

    def test_class_of_an_unknown_type_is_none_not_a_guess(self):
        assert TypeClassIndex(_forcefield()).class_of("not_a_type") is None

    def test_layer_of_takes_the_highest_layer_in_the_pattern(self):
        index = TypeClassIndex(_forcefield())

        assert index.layer_of(("CT", "CT")) == 0
        assert index.layer_of(("CT", "CR")) == 1  # the CL&P overlay wins

    def test_layer_of_an_unknown_class_is_zero(self):
        assert TypeClassIndex(_forcefield()).layer_of(("nothing",)) == 0


class TestScore:
    """Higher is more specific; ``None`` means the pattern does not match at all."""

    def test_exact_type_outscores_class_which_outscores_wildcard(self):
        index = TypeClassIndex(_forcefield())
        atoms = ["opls_135", "opls_140"]

        exact = index.score(("opls_135", "opls_140"), atoms)
        by_class = index.score(("CT", "HC"), atoms)
        wildcard = index.score(("*", "*"), atoms)

        assert exact > by_class > wildcard
        assert wildcard == 0

    def test_a_component_matching_nothing_kills_the_pattern(self):
        index = TypeClassIndex(_forcefield())

        assert index.score(("opls_135", "opls_999"), ["opls_135", "opls_140"]) is None

    def test_a_pattern_matches_end_for_end_reversed(self):
        index = TypeClassIndex(_forcefield())
        pattern = ("opls_135", "opls_140")

        forward = index.score(pattern, ["opls_135", "opls_140"])
        reverse = index.score(pattern, ["opls_140", "opls_135"])

        assert forward == reverse

    def test_a_wildcard_ended_dihedral_scores_below_a_resolved_one(self):
        """X-CT-CT-X must lose to opls_135-CT-CT-opls_135. This is why OPLS works."""
        index = TypeClassIndex(_forcefield())
        atoms = ["opls_135"] * 4

        wildcarded = index.score(("*", "CT", "CT", "*"), atoms)
        resolved = index.score(("opls_135", "CT", "CT", "opls_135"), atoms)

        assert resolved > wildcarded

    def test_score_requires_matching_arity(self):
        """A 3-component pattern is never silently truncated onto a 2-atom term."""
        index = TypeClassIndex(_forcefield())

        with pytest.raises(ValueError):
            index.score(("CT", "CT", "CT"), ["opls_135", "opls_135"])
