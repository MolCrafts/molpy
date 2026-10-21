"""CGsmiles notation — ``molrs::io::cgsmiles``.

CGsmiles writes a molecule at one or more *coarse-grained* resolutions — a
resolution at which one particle, a *bead*, stands in for a whole group of
atoms. :func:`read_cgsmiles` reads a string; ``to_atomistic()`` expands its
lowest level into atoms (topology only — atoms, bonds and the per-atom
``frag_id`` saying which bead each atom came from), ``templates()`` builds one
ported template per fragment, and ``to_coarsegrain()`` reads the coarsest
level as a bead graph. A line notation states no geometry, so coordinates,
hydrogens and perception remain separate steps.

:class:`CgSmilesIr` is the parsed string, handed out by :func:`read_cgsmiles`
and never constructed by the caller: one :class:`CgGraph` per resolution
level plus the fragment tables that resolve them out. The records it hands
out — :class:`CgGraph`, :class:`CgNode`, :class:`CgEdge`,
:class:`CgFragmentDef` (whose atomistic body is a
:class:`molrs.io.smiles.FragmentIr`), :class:`ResolvedPair` and
:class:`PairEnd` — are read-only views over the parsed value, so no fact of
the notation has to be re-parsed, decoded or unpacked from a bare tuple
position on the Python side. The enums the notation does not spell out keep
lowercase variant names: ``ResolvedPair.kind`` is a bond kind (``"single"``,
``"aromatic"``, …) and ``PairEnd.end`` is ``"sub"`` or ``"body"``. A refusal
is a :class:`molrs.io.smiles.SmilesError` with ``notation == "cgsmiles"``.

There is no ``CgSmilesReader``, deliberately: a parser of one string into an
IR is not an indexed frame cursor.
"""

from .._native import (
    CgEdge,
    CgFragmentDef,
    CgGraph,
    CgNode,
    CgSmilesIr,
    PairEnd,
    ResolvedPair,
    read_cgsmiles,
)

__all__ = [
    "CgEdge",
    "CgFragmentDef",
    "CgGraph",
    "CgNode",
    "CgSmilesIr",
    "PairEnd",
    "ResolvedPair",
    "read_cgsmiles",
]
