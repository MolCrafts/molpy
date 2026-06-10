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

from molrs.fields import *  # noqa: F401,F403  (re-export canonical registry)
from molrs.fields import FieldFormatter, FieldSpec, __all__ as _MOLRS_FIELDS_ALL

# ===================================================================
#                    ForceFieldFormatter
# ===================================================================


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

    def format_params(self, typ: object, style: object) -> list[float]:
        """Dispatch to the registered formatter for *style*'s class.

        Args:
            typ: A Type object (BondType, AngleType, etc.)
            style: The Style object that contains *typ*.

        Returns:
            Parameters in the target format's order.

        Raises:
            ValueError: If no formatter is registered for the style class.
        """
        style_class = type(style)
        if style_class in self._param_formatters:
            formatter = self._param_formatters[style_class]
            try:
                return formatter(typ)
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to format parameters for {style_class.__name__} "
                    f"with type {type(typ).__name__}: {e}"
                ) from e
        raise ValueError(
            f"No param formatter registered for {style_class.__name__}. "
            f"Available: {[c.__name__ for c in self._param_formatters]}"
        )


__all__ = [*_MOLRS_FIELDS_ALL, "ForceFieldFormatter"]
