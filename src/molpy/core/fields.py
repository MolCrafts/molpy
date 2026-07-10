"""Canonical field registry — re-exported from molrs (single source of truth).

The canonical field names and :class:`FieldSpec`/:class:`FieldFormatter` types now
live in :mod:`molrs.fields`, whose keys are sourced from the Rust ``molrs.keys``
constants. molpy re-exports them wholesale so existing
``from molpy.core.fields import …`` call sites keep resolving, and adds only the
force-field-specific :class:`ForceFieldFormatter` on top (until FF I/O is sunk
into molrs-io).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from molrs.fields import *  # noqa: F401,F403  (re-export canonical registry)
from molrs.fields import FieldFormatter, FieldSpec, __all__ as _MOLRS_FIELDS_ALL

# ===================================================================
#                    molpy-owned canonical fields
# ===================================================================
# Assembly is a molpy concept, so its field lives here rather than in the molrs
# registry. It is a real ``FieldSpec``, not a ``site_field: str`` constructor
# knob: field names are never strings passed around.

#: Reaction-site label on an atom. Sparse: only the atoms a reaction may bind
#: carry a name (``"a"``, ``"b"``, …); every other atom holds the empty string,
#: which means *unmarked* and never matches a ``%site`` predicate. A missing
#: ``site`` column is an error, not "no sites".
SITE = FieldSpec(
    key="site",
    dtype=np.dtype("U16"),
    doc="Assembly site label; empty string means the atom is not a site.",
)

# ===================================================================
#                    ForceFieldFormatter
# ===================================================================

# Category name for each base molrs style class, used to classify formatter
# registry keys. A specialized style class additionally pins a style ``name``
# (read by instantiating it with no arguments); a base style class leaves the
# name unresolved (``None``) and acts as the category-wide fallback.
_BASE_STYLE_CATEGORIES = {
    "AtomStyle": "atom",
    "BondStyle": "bond",
    "AngleStyle": "angle",
    "DihedralStyle": "dihedral",
    "ImproperStyle": "improper",
    "PairStyle": "pair",
}

_STYLE_IDENTITY_CACHE: dict[type, tuple[str | None, str | None]] = {}


def _style_class_identity(style_class: type) -> tuple[str | None, str | None]:
    """Return ``(category, name)`` identifying a Style class.

    Base category styles (``BondStyle``…) map to ``(category, None)``; a
    specialized style maps to its fixed ``(category, name)`` by instantiating
    it with no arguments. Results are memoised on the class.
    """
    cached = _STYLE_IDENTITY_CACHE.get(style_class)
    if cached is not None:
        return cached

    if style_class.__name__ in _BASE_STYLE_CATEGORIES:
        identity: tuple[str | None, str | None] = (
            _BASE_STYLE_CATEGORIES[style_class.__name__],
            None,
        )
    else:
        try:
            instance = style_class()
            identity = (
                getattr(instance, "category", None),
                getattr(instance, "name", None),
            )
        except Exception:
            identity = (None, None)

    _STYLE_IDENTITY_CACHE[style_class] = identity
    return identity


class ForceFieldFormatter(FieldFormatter):
    """Extends :class:`FieldFormatter` with force-field parameter formatting.

    Inherits all field-name mapping from ``FieldFormatter``. Adds a
    ``_param_formatters`` registry that maps Style classes to serialization
    functions, following the same per-subclass dispatch pattern.

    Example::

        class LammpsForceFieldFormatter(LammpsFieldFormatter, ForceFieldFormatter):
            _param_formatters = {
                BondHarmonicStyle: _format_bond_harmonic,
                AngleHarmonicStyle: _format_angle_harmonic,
            }
    """

    _param_formatters: dict[type, Callable] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._param_formatters = dict(cls._param_formatters)

    @classmethod
    def register_param_formatter(cls, style_class: type, fn: Callable) -> None:
        """Register a param formatter at runtime."""
        cls._param_formatters[style_class] = fn

    def _resolve_formatter(self, style: object) -> Callable | None:
        """Resolve the registered formatter for *style*.

        molrs returns styles as their base category class (``BondStyle``,
        ``PairStyle``, …) regardless of which named/specialized style was
        registered, so an exact ``type(style)`` match only catches the generic
        fallbacks. Specialized formatters are therefore also matched by the
        style's ``(category, name)`` against each registered Style class —
        whose own ``category``/``name`` identify it (e.g. a ``BondHarmonicStyle``
        instance has ``category == "bond"`` and ``name == "harmonic"``).
        """
        formatters = self._param_formatters

        style_category = getattr(style, "category", None)
        style_name = getattr(style, "name", None)
        if style_category is None or style_name is None:
            # No molrs-style identity to match on; fall back to an exact
            # class match (legacy / non-molrs styles).
            return formatters.get(type(style))

        # molrs returns every style as its base category class regardless of
        # which named/specialized style was registered, so ``type(style)`` is
        # useless for dispatch. Match by the style's ``(category, name)`` against
        # each registered Style class's own identity; a base category class
        # (name ``None``) is the category-wide fallback, a specialized class with
        # a matching name wins.
        specialized: Callable | None = None
        generic: Callable | None = None
        for style_class, fn in formatters.items():
            key_category, key_name = _style_class_identity(style_class)
            if key_category != style_category:
                continue
            if key_name is None:
                generic = fn
            elif key_name == style_name:
                specialized = fn
        return specialized or generic

    def format_params(self, typ: object, style: object) -> list[float]:
        """Dispatch to the registered formatter for *style*.

        Args:
            typ: A Type object (BondType, AngleType, etc.)
            style: The Style object that contains *typ*.

        Returns:
            Parameters in the target format's order.

        Raises:
            ValueError: If no formatter is registered for the style.
        """
        formatter = self._resolve_formatter(style)
        if formatter is not None:
            try:
                return formatter(typ)
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to format parameters for style {getattr(style, 'name', style)!r} "
                    f"with type {type(typ).__name__}: {e}"
                ) from e
        raise ValueError(
            f"No param formatter registered for style {getattr(style, 'name', style)!r}. "
            f"Available: {[c.__name__ for c in self._param_formatters]}"
        )


__all__ = [*_MOLRS_FIELDS_ALL, "ForceFieldFormatter", "SITE"]
