"""Chemical perception — ``molrs::perceive``.

*Perception* derives chemical facts a connectivity graph implies but does not
state. One layer above ``core`` and below ``ff`` / ``io`` / ``conformer``:
rings (SSSR, the smallest set of smallest rings), aromaticity, hydrogens,
stereochemistry, rotatable bonds, SMARTS (SMILES Arbitrary Target
Specification) substructure matching, and coarse-grained bead-pattern
matching. Gasteiger charges are a charge model and live in :mod:`molrs.ff`.

Every perception is a free function with the Rust name, in two shapes:
``perceive_<fact>(mol)`` is a *query* — it reports a side table
(:func:`perceive_rings` → :class:`RingSet`, :func:`perceive_rotatable_bonds`
→ ``list[RotatableBond]``) and never touches the molecule;
``assign_<fact>(mol)`` is a *mutation* — it writes the fact onto ``mol``
itself, in place, as atom / bond props, and returns ``None``
(:func:`assign_kekule_bond_orders` returns whether every aromatic bond got a
legal number). To keep the input, copy it first: ``copy.copy(mol)``.
:func:`add_hydrogens` is a graph edit and keeps its verb: it returns a new
graph. Atoms in results are the molecule's own views.

:class:`BondOrderSource` and :class:`UnknownBondPolicy` mirror the core enums
of the same names: where :func:`assign_bcc_bond_types` takes bond orders from,
and what an unclassified bond counts as for rotatable-bond perception.

SMARTS lives here because a pattern is a query over a *perceived* graph —
matching needs ring membership and aromaticity, not a text format. The SMILES
front-end is a format, and lives in :mod:`molrs.io`.

:class:`Reaction` applies a reaction SMARTS (SMIRKS) to a molecule: the
transform is defined by a pattern over the perceived graph, so it is
perception's too; :meth:`Reaction.apply_many` returns one
:class:`ReactionBatch`.

:class:`SubgraphMatcher` is the coarse-grained counterpart: it snapshots a bead
pattern (a :class:`~molrs.core.CoarseGrain`, e.g. from
``read_cgsmiles(...).to_coarsegrain()``) and lists every occurrence of it in a
target ``CoarseGrain`` as groups of bead views. It does not partition
overlapping groups. Mapping matched groups onto the sites of a new
``CoarseGrain`` is construction, not perception: that is
:class:`molrs.builder.Coarsener`.
"""

from ._native import (
    BondOrderSource,
    Reaction,
    ReactionBatch,
    RingSet,
    RotatableBond,
    SmartsMatch,
    SmartsPattern,
    SubgraphMatcher,
    UnknownBondPolicy,
    add_hydrogens,
    assign_aromaticity,
    assign_bcc_bond_types,
    assign_bond_orders,
    assign_equivalence_classes,
    assign_kekule_bond_orders,
    assign_rings,
    assign_rotatable_bonds,
    assign_stereo,
    perceive_rings,
    perceive_rotatable_bonds,
)

__all__ = [
    "BondOrderSource",
    "Reaction",
    "ReactionBatch",
    "RingSet",
    "RotatableBond",
    "SmartsMatch",
    "SmartsPattern",
    "SubgraphMatcher",
    "UnknownBondPolicy",
    "add_hydrogens",
    "assign_aromaticity",
    "assign_bcc_bond_types",
    "assign_bond_orders",
    "assign_equivalence_classes",
    "assign_kekule_bond_orders",
    "assign_rings",
    "assign_rotatable_bonds",
    "assign_stereo",
    "perceive_rings",
    "perceive_rotatable_bonds",
]
