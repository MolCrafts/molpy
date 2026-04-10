"""Canonical field definitions and I/O boundary formatters.

This module defines:

- ``FieldSpec``: canonical field metadata (key, dtype, shape, doc).
- ``FieldFormatter``: base class for per-format field name translation.
- ``ForceFieldFormatter``: extends ``FieldFormatter`` with FF parameter formatting.

Architecture::

    FieldSpec                         — canonical field definition
        ↓
    FieldFormatter                    — data field mapping registry
        ↓                               canonicalize() / localize() on Block
    ForceFieldFormatter(FieldFormatter) — inherits field mapping + adds param formatters

Each I/O format defines its own ``FieldFormatter`` subclass in its own module.
ForceField writers inherit from the corresponding data formatter to share
field mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from molpy.core.frame import Block, Frame

# ===================================================================
#                       FieldSpec
# ===================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Specification of a canonical field in the internal data model.

    Attributes:
        key: Canonical field name used internally.
        dtype: NumPy dtype for this field's array.
        shape: Dimensionality hint (``"scalar"``, ``"vector3"``, ...).
        doc: Human-readable description with unit.
    """

    key: str
    dtype: np.dtype
    shape: str
    doc: str


# ===================================================================
#               Canonical Atom Fields
# ===================================================================

ATOM_ID = FieldSpec("id", np.dtype(np.int64), "scalar", "atom ID (1-indexed)")
ATOM_TYPE = FieldSpec("type", np.dtype("U64"), "scalar", "force field type label")
CHARGE = FieldSpec("charge", np.dtype(np.float64), "scalar", "partial charge (e)")
MASS = FieldSpec("mass", np.dtype(np.float64), "scalar", "atomic mass (amu)")
MOL_ID = FieldSpec("mol_id", np.dtype(np.int64), "scalar", "molecule ID (1-indexed)")
ELEMENT = FieldSpec("element", np.dtype("U4"), "scalar", "element symbol")
POS_X = FieldSpec("x", np.dtype(np.float64), "scalar", "x coordinate (Angstrom)")
POS_Y = FieldSpec("y", np.dtype(np.float64), "scalar", "y coordinate (Angstrom)")
POS_Z = FieldSpec("z", np.dtype(np.float64), "scalar", "z coordinate (Angstrom)")
VEL_X = FieldSpec("vx", np.dtype(np.float64), "scalar", "x velocity (Angstrom/fs)")
VEL_Y = FieldSpec("vy", np.dtype(np.float64), "scalar", "y velocity (Angstrom/fs)")
VEL_Z = FieldSpec("vz", np.dtype(np.float64), "scalar", "z velocity (Angstrom/fs)")
RES_ID = FieldSpec("res_id", np.dtype(np.int64), "scalar", "residue ID")
RES_NAME = FieldSpec("res_name", np.dtype("U8"), "scalar", "residue name")

# ===================================================================
#               Canonical Bond Fields
# ===================================================================

BOND_TYPE = FieldSpec("type", np.dtype("U64"), "scalar", "bond type label")
BOND_ATOMI = FieldSpec(
    "atomi", np.dtype(np.int64), "scalar", "first atom index (0-indexed)"
)
BOND_ATOMJ = FieldSpec(
    "atomj", np.dtype(np.int64), "scalar", "second atom index (0-indexed)"
)

# ===================================================================
#               Canonical Angle Fields
# ===================================================================

ANGLE_TYPE = FieldSpec("type", np.dtype("U64"), "scalar", "angle type label")
ANGLE_ATOMI = FieldSpec(
    "atomi", np.dtype(np.int64), "scalar", "first atom index (0-indexed)"
)
ANGLE_ATOMJ = FieldSpec(
    "atomj", np.dtype(np.int64), "scalar", "vertex atom index (0-indexed)"
)
ANGLE_ATOMK = FieldSpec(
    "atomk", np.dtype(np.int64), "scalar", "third atom index (0-indexed)"
)

# ===================================================================
#               Canonical Dihedral Fields
# ===================================================================

DIHEDRAL_TYPE = FieldSpec("type", np.dtype("U64"), "scalar", "dihedral type label")
DIHEDRAL_ATOMI = FieldSpec(
    "atomi", np.dtype(np.int64), "scalar", "first atom index (0-indexed)"
)
DIHEDRAL_ATOMJ = FieldSpec(
    "atomj", np.dtype(np.int64), "scalar", "second atom index (0-indexed)"
)
DIHEDRAL_ATOMK = FieldSpec(
    "atomk", np.dtype(np.int64), "scalar", "third atom index (0-indexed)"
)
DIHEDRAL_ATOML = FieldSpec(
    "atoml", np.dtype(np.int64), "scalar", "fourth atom index (0-indexed)"
)


# ===================================================================
#                       FieldFormatter
# ===================================================================


class FieldFormatter:
    """Translates between format-specific and canonical field names.

    Subclasses define ``_field_formatters`` as a class-level registry mapping
    format-native column names to :class:`FieldSpec` objects.  Registrations
    are isolated per subclass via ``__init_subclass__``.

    Example::

        class LammpsFieldFormatter(FieldFormatter):
            _field_formatters = {
                "q":   CHARGE,   # LAMMPS "q" → canonical "charge"
                "mol": MOL_ID,   # LAMMPS "mol" → canonical "mol_id"
            }
    """

    _field_formatters: dict[str, FieldSpec] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._field_formatters = dict(cls._field_formatters)

    @classmethod
    def register_field(cls, format_key: str, spec: FieldSpec) -> None:
        """Register a field mapping at runtime."""
        cls._field_formatters[format_key] = spec

    # ── Block-level translation ──────────────────────────────────

    def canonicalize(self, block: Block) -> Block:
        """Reader exit: rename format-specific keys to canonical."""
        for fmt_key, spec in self._field_formatters.items():
            if fmt_key in block and spec.key not in block:
                block.rename(fmt_key, spec.key)
        return block

    def localize(self, block: Block) -> Block:
        """Writer entry: rename canonical keys to format-specific."""
        for fmt_key, spec in self._field_formatters.items():
            if spec.key in block and fmt_key not in block:
                block.rename(spec.key, fmt_key)
        return block

    # ── Frame-level convenience ──────────────────────────────────

    def canonicalize_frame(self, frame: Frame) -> Frame:
        """Canonicalize all blocks in a Frame (in-place)."""
        for key in list(frame._blocks):
            self.canonicalize(frame[key])
        return frame

    def localize_frame(self, frame: Frame) -> Frame:
        """Localize all blocks in a Frame copy (non-destructive)."""
        frame = frame.copy()
        for key in list(frame._blocks):
            self.localize(frame[key])
        return frame


# ===================================================================
#                    ForceFieldFormatter
# ===================================================================


class ForceFieldFormatter(FieldFormatter):
    """Extends :class:`FieldFormatter` with force-field parameter formatting.

    Inherits all field-name mapping from ``FieldFormatter``.  Adds a
    ``_param_formatters`` registry that maps Style classes to serialization
    functions, following the same dispatch pattern.

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
