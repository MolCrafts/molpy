"""The force-field data model — ``molrs::ff::forcefield``.

:class:`ForceField` holds styles (one per category and style name, in the
force-field IR, which adopts the LAMMPS standard) and the types defined under
them; :class:`Style` / :class:`ForceFieldType` and their per-category subclasses are
live handles onto it.

No file format is here. Every force-field file reader and writer —
``read_lammps_forcefield``, ``read_top_forcefield``,
``write_frcmod``, ``write_openmm_forcefield`` and the rest — is a function at the top of
:mod:`molrs.io`, as every other file reader and writer is.
"""

from .._native import (
    AngleStyle,
    AngleType,
    AtomStyle,
    AtomType,
    BondStyle,
    BondType,
    CmapStyle,
    CmapType,
    DihedralStyle,
    DihedralType,
    ForceField,
    ImproperStyle,
    ImproperType,
    PairStyle,
    PairType,
    RelationStyle,
    RelationType,
    Style,
    ForceFieldType,
)

__all__ = [
    "AngleStyle",
    "AngleType",
    "AtomStyle",
    "AtomType",
    "BondStyle",
    "BondType",
    "CmapStyle",
    "CmapType",
    "DihedralStyle",
    "DihedralType",
    "ForceField",
    "ImproperStyle",
    "ImproperType",
    "PairStyle",
    "PairType",
    "RelationStyle",
    "RelationType",
    "Style",
    "ForceFieldType",
]
