"""Canonical field names and I/O name translation (single source of truth).

Canonical column names are **plain strings** sourced from the native ``keys``
table and re-exported here one by one (``fields.CHARGE == "charge"``), so an
annotation dict unpacks as ``update(**kwargs)`` and ``d["type"]`` keeps working.

Name translation at a format boundary is :class:`FieldFormatter`, also native:
a subclass maps ``{format_key: canonical_key}`` in ``_field_formatters`` and
applies it with ``canonicalize``/``localize`` (per Block) or the ``*_frame``
variants; ``register_field`` adds a mapping at runtime and ``__init_subclass__``
keeps each subclass's registry isolated. The formatters for the formats the core
parses (LAMMPS, MOL2, PDB, XYZ) are native and re-exported here; a format
molpy parses itself declares its own subclass in its I/O module.

Both the canonical names and the ``FieldFormatter`` family are native
re-exports; molpy adds nothing here.
"""

from __future__ import annotations

from molrs import keys as _keys
from molrs.fields import (
    FieldFormatter,
    LammpsFieldFormatter,
    Mol2FieldFormatter,
    PdbFieldFormatter,
    XyzFieldFormatter,
)

# Canonical column names come from molrs — one table, projected. Re-exported
# as plain strings so annotation dicts unpack as ``update(**kwargs)`` and
# ``d["type"]`` keeps working. ``molrs.keys.Key`` is still accepted as a
# Block / Atom column address; it is not a valid Python keyword name. The
# list is spelled out (not built from ``dir(molrs.keys)``) so a type checker
# can resolve ``fields.CHARGE``.
ATOMI = _keys.ATOMI.key
ATOMIC_NUMBER = _keys.ATOMIC_NUMBER.key
ATOMJ = _keys.ATOMJ.key
ATOMK = _keys.ATOMK.key
ATOML = _keys.ATOML.key
BEAD_TYPE = _keys.BEAD_TYPE.key
BOND_NUMBER = _keys.BOND_NUMBER.key
BOND_TYPE = _keys.BOND_TYPE.key
CHARGE = _keys.CHARGE.key
ELEMENT = _keys.ELEMENT.key
EXCLUDE_14 = _keys.EXCLUDE_14.key
ID = _keys.ID.key
IS_14 = _keys.IS_14.key
IX = _keys.IX.key
IY = _keys.IY.key
IZ = _keys.IZ.key
MASS = _keys.MASS.key
MOL_ID = _keys.MOL_ID.key
MUX = _keys.MUX.key
MUY = _keys.MUY.key
MUZ = _keys.MUZ.key
NAME = _keys.NAME.key
QUATI = _keys.QUATI.key
QUATJ = _keys.QUATJ.key
QUATK = _keys.QUATK.key
QUATW = _keys.QUATW.key
RES_ID = _keys.RES_ID.key
RES_NAME = _keys.RES_NAME.key
TYPE = _keys.TYPE.key
TYPE_ID = _keys.TYPE_ID.key
VX = _keys.VX.key
VY = _keys.VY.key
VZ = _keys.VZ.key
X = _keys.X.key
Y = _keys.Y.key
Z = _keys.Z.key
# Column groups (plain lists of names), not single keys.
COORDS = _keys.COORDS
DIPOLE = _keys.DIPOLE
ENDPOINTS = _keys.ENDPOINTS
QUAT = _keys.QUAT
VELOCITIES = _keys.VELOCITIES

__all__ = [
    "FieldFormatter",
    "LammpsFieldFormatter",
    "Mol2FieldFormatter",
    "PdbFieldFormatter",
    "XyzFieldFormatter",
]
