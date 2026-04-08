"""Unit tests for reaction preset registry."""

import pytest

from molpy.builder.polymer.presets import (
    ReactionPresetSpec,
    ReactionPresets,
)
from molpy.reacter.base import Reacter
from molpy.reacter.selectors import select_hydrogens, select_self
from molpy.reacter.utils import form_single_bond


class TestReactionPresetSpec:
    def test_frozen(self):
        spec = ReactionPresetSpec(
            name="test",
            description="test desc",
            site_selector_left=select_self,
            site_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )
        with pytest.raises(AttributeError):
            spec.name = "changed"

    def test_fields(self):
        spec = ReactionPresetSpec(
            name="my_preset",
            description="My preset",
            site_selector_left=select_self,
            site_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )
        assert spec.name == "my_preset"
        assert spec.description == "My preset"


class TestReactionPresets:
    def test_builtins_registered(self):
        presets = ReactionPresets.list_presets()
        assert "dehydration" in presets
        assert "condensation" in presets

    def test_get_dehydration(self):
        reacter = ReactionPresets.get("dehydration")
        assert isinstance(reacter, Reacter)
        assert reacter.name == "dehydration"

    def test_get_condensation(self):
        reacter = ReactionPresets.get("condensation")
        assert isinstance(reacter, Reacter)
        assert reacter.name == "condensation"

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown reaction preset"):
            ReactionPresets.get("nonexistent_preset")

    def test_get_spec(self):
        spec = ReactionPresets.get_spec("dehydration")
        assert isinstance(spec, ReactionPresetSpec)
        assert spec.name == "dehydration"
        assert spec.site_selector_left is select_self
        assert spec.bond_former is form_single_bond

    def test_register_custom(self):
        custom_spec = ReactionPresetSpec(
            name="_test_custom_preset",
            description="Custom test preset",
            site_selector_left=select_self,
            site_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )
        ReactionPresets.register(custom_spec)
        assert "_test_custom_preset" in ReactionPresets.list_presets()

        reacter = ReactionPresets.get("_test_custom_preset")
        assert isinstance(reacter, Reacter)
        assert reacter.name == "_test_custom_preset"

        # Cleanup: remove from registry to avoid polluting other tests
        del ReactionPresets._presets["_test_custom_preset"]

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            ReactionPresets.register(
                ReactionPresetSpec(
                    name="dehydration",
                    description="duplicate",
                    site_selector_left=select_self,
                    site_selector_right=select_self,
                    leaving_selector_left=select_hydrogens(1),
                    leaving_selector_right=select_hydrogens(1),
                    bond_former=form_single_bond,
                )
            )

    def test_list_presets_sorted(self):
        presets = ReactionPresets.list_presets()
        assert presets == sorted(presets)

    def test_get_returns_new_reacter_each_time(self):
        r1 = ReactionPresets.get("dehydration")
        r2 = ReactionPresets.get("dehydration")
        assert r1 is not r2
