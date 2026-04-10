"""Reaction preset registry for common polymerization reactions.

Provides named presets that bundle site selectors, leaving selectors,
and bond formers into ready-to-use Reacter instances.

Built-in presets:
- ``"dehydration"`` — remove one H from each site (PEO, general)
- ``"condensation"`` — site via C neighbor, remove one H each side
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from molpy.reacter.base import BondFormer, Reacter
from molpy.reacter.selectors import Selector


@dataclass(frozen=True)
class ReactionPresetSpec:
    """Immutable specification for a reaction preset.

    Attributes:
        name: Unique preset name
        description: Human-readable description
        site_selector_left: Maps left port atom to site atom
        site_selector_right: Maps right port atom to site atom
        leaving_selector_left: Identifies left leaving group
        leaving_selector_right: Identifies right leaving group
        bond_former: Creates bond between site atoms
    """

    name: str
    description: str
    site_selector_left: Selector
    site_selector_right: Selector
    leaving_selector_left: Selector
    leaving_selector_right: Selector
    bond_former: BondFormer


class ReactionPresets:
    """Class-level registry of named reaction presets.

    Example::

        reacter = ReactionPresets.get("dehydration")
        # reacter is ready to use with PolymerBuilder
    """

    _presets: ClassVar[dict[str, ReactionPresetSpec]] = {}

    @classmethod
    def register(cls, spec: ReactionPresetSpec) -> None:
        """Register a reaction preset.

        Args:
            spec: Preset specification to register.

        Raises:
            ValueError: If a preset with the same name is already registered.
        """
        if spec.name in cls._presets:
            raise ValueError(
                f"Preset {spec.name!r} is already registered. "
                "Use a different name or unregister the existing one first."
            )
        cls._presets[spec.name] = spec

    @classmethod
    def get(cls, name: str) -> Reacter:
        """Create a Reacter instance from a named preset.

        Args:
            name: Preset name (e.g. ``"dehydration"``).

        Returns:
            Configured Reacter instance.

        Raises:
            KeyError: If preset name is not registered.
        """
        if name not in cls._presets:
            available = sorted(cls._presets.keys())
            raise KeyError(
                f"Unknown reaction preset: {name!r}. Available presets: {available}"
            )
        spec = cls._presets[name]
        return Reacter(
            name=spec.name,
            anchor_selector_left=spec.site_selector_left,
            anchor_selector_right=spec.site_selector_right,
            leaving_selector_left=spec.leaving_selector_left,
            leaving_selector_right=spec.leaving_selector_right,
            bond_former=spec.bond_former,
        )

    @classmethod
    def list_presets(cls) -> list[str]:
        """Return sorted list of registered preset names."""
        return sorted(cls._presets.keys())

    @classmethod
    def get_spec(cls, name: str) -> ReactionPresetSpec:
        """Return the raw spec for a preset without creating a Reacter.

        Args:
            name: Preset name.

        Raises:
            KeyError: If preset name is not registered.
        """
        if name not in cls._presets:
            available = sorted(cls._presets.keys())
            raise KeyError(
                f"Unknown reaction preset: {name!r}. Available presets: {available}"
            )
        return cls._presets[name]


# ============================================================================
# Built-in presets
# ============================================================================


def _register_builtins() -> None:
    """Register built-in reaction presets."""
    from molpy.reacter.selectors import (
        select_hydrogens,
        select_neighbor,
        select_self,
    )
    from molpy.reacter.utils import form_single_bond

    # Dehydration: port atom IS the site, remove 1 H from each side
    ReactionPresets.register(
        ReactionPresetSpec(
            name="dehydration",
            description="Dehydration condensation: remove one H from each site, form single bond",
            site_selector_left=select_self,
            site_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )
    )

    # Condensation: site is the C neighbor of the port atom
    ReactionPresets.register(
        ReactionPresetSpec(
            name="condensation",
            description="Condensation: site via C neighbor, remove one H from each side",
            site_selector_left=select_neighbor("C"),
            site_selector_right=select_self,
            leaving_selector_left=select_hydrogens(1),
            leaving_selector_right=select_hydrogens(1),
            bond_former=form_single_bond,
        )
    )


_register_builtins()
