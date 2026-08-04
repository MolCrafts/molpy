"""Chemical perception, returning molpy graphs.

:class:`molrs.perceive.Perceive` is the owner of every perception step — rings,
aromaticity, hydrogens, stereo, rotatable bonds, bond types, equivalence
classes — as a builder: graph in, graph out, nothing mutated. molpy adds one
thing and only one: the graph that comes back is a molpy
:class:`~molpy.core.atomistic.Atomistic`, so callers never write
``Atomistic.adopt(...)`` around a perception result.

That is the whole class. Every method is ``super()`` plus an adopt; there is no
molpy-side perception logic, and there must not be — see
``.claude/notes/architecture.md`` on the thin-inheritance layer.
"""

from __future__ import annotations

import molrs

from molpy.core.atomistic import Atomistic

__all__ = ["Perceive"]


class Perceive(molrs.perceive.Perceive):
    """molrs' perception builder, handing back molpy graphs.

    Non-mutating throughout: each method returns a new graph and leaves its
    argument untouched.

    Examples:
        >>> import molpy as mp
        >>> mol = mp.io.read_smiles("c1ccccc1")
        >>> filled = mp.Perceive().find_hydrogens(mol)
        >>> len(list(filled.atoms))
        12
    """

    def find_rings(self, mol: Atomistic) -> Atomistic:
        """Annotate ring membership. To *query* rings instead, use `RingInfo`."""
        return Atomistic.adopt(super().find_rings(mol))

    def find_aromaticity(self, mol: Atomistic) -> Atomistic:
        """Bring the graph to the standard aromatic representation.

        Aromatic atoms get ``is_aromatic``, aromatic bonds get
        ``bond_type = 4``, and every bond gets an integer ``bond_number`` — the
        localized Lewis structure. Nothing carries a fractional order, because
        aromaticity is a bond *type*.

        Hydrogens are neither added nor needed: implicit hydrogens are read off
        each atom's valence, so `find_hydrogens` stays independent.
        """
        return Atomistic.adopt(super().find_aromaticity(mol))

    def find_hydrogens(self, mol: Atomistic) -> Atomistic:
        """Add the hydrogens implied by each heavy atom's open valence."""
        return Atomistic.adopt(super().find_hydrogens(mol))

    def find_stereo(self, mol: Atomistic) -> Atomistic:
        """Annotate perceived stereochemistry."""
        return Atomistic.adopt(super().find_stereo(mol))

    def find_rotatable(self, mol: Atomistic) -> Atomistic:
        """Annotate rotatable bonds."""
        return Atomistic.adopt(super().find_rotatable(mol))

    def find_bond_types(self, mol: Atomistic) -> Atomistic:
        """Annotate the perceived AM1-BCC bond type of every bond."""
        return Atomistic.adopt(super().find_bond_types(mol))

    def find_kekule_orders(self, mol: Atomistic) -> Atomistic:
        """Give every aromatic bond a legal localized ``bond_number``.

        Kekulization only — a graph whose aromatic bonds are not marked yet
        comes back unchanged, because deciding *which* bonds are aromatic is
        `find_aromaticity`'s job.
        """
        return Atomistic.adopt(super().find_kekule_orders(mol))

    def find_equivalence_classes(self, mol: Atomistic) -> Atomistic:
        """Annotate topological equivalence classes."""
        return Atomistic.adopt(super().find_equivalence_classes(mol))
