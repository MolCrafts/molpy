"""The force-field data model — ``molrs::ff::forcefield``.

:class:`ForceField` holds styles (one per category and style name, in the
force-field IR, which adopts the LAMMPS standard) and the types defined under
them; :class:`Style` / :class:`ForceFieldType` and their per-category subclasses are
live handles onto it.

No file format is here. Every force-field file reader and writer is a class
of its format's :mod:`molrs.io` submodule —
:class:`molrs.io.lammps.LammpsForcefieldReader`,
:class:`molrs.io.gromacs.GromacsTopForcefieldReader`,
:class:`molrs.io.amber.AmberFrcmodWriter`,
:class:`molrs.io.openmm_xml.OpenmmXmlWriter` and the rest — as every other
file reader and writer is.
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
