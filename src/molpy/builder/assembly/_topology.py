"""Pair adjacent residues — the rule a polymer builder uses.

A repeat unit is a residue. Once :class:`~molpy.builder.assembly._library.MonomerLibrary`
has stamped ``RES_ID`` onto each pasted copy, "bond monomer *i* to monomer *i+1*"
is a lookup, not a search: this selector indexes the matched sites by residue and
walks the topology's edges. There is no O(sites²) pairing here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from molpy.builder.assembly._selector import Binding, Selector
from molpy.core import fields

if TYPE_CHECKING:
    from molpy.builder.assembly._context import MatchContext
    from molpy.parser.smiles.cgsmiles_ir import CGSmilesGraphIR


class TopologySelector(Selector):
    """Bond the sites of residues that the topology says are adjacent.

    Each edge ``(i, j)`` of the topology consumes residue ``i``'s site on the
    reaction's first reactant and residue ``j``'s site on the second. A linear
    chain is a topology whose edges form a path; a four-arm star is a residue
    with four sites; a macrocycle is one more edge between residues that are
    already connected. Nothing here knows which of those it is building.
    """

    def __init__(self, topology: CGSmilesGraphIR) -> None:
        self._topology = topology

    @staticmethod
    def residue_ids(topology: CGSmilesGraphIR) -> dict[int, int]:
        """Map each topology node onto a 1-based residue id, in notation order.

        The parser's node ids are an internal counter and need not be contiguous;
        a residue id reaches a PDB or a prmtop, so it must be. Both the library
        that stamps ``RES_ID`` and the selector that reads it derive the numbering
        from the same topology, so they cannot disagree.
        """
        return {node.id: index for index, node in enumerate(topology.nodes, start=1)}

    def select(self, context: MatchContext) -> Iterator[Binding]:
        sites_a = self._index(context, context.comp_a, context.map_a)
        sites_b = self._index(context, context.comp_b, context.map_b)
        residue_of = self.residue_ids(self._topology)
        used: set[int] = set()

        for bond in self._topology.bonds:
            i, j = residue_of[bond.node_i.id], residue_of[bond.node_j.id]
            # An edge is undirected, but the reaction is not: one residue must
            # supply the first reactant's site and the other the second. Which is
            # which is decided by what is still free, not by the edge's direction
            # — otherwise a ring-closure bond would demand a site the opening
            # bond already consumed.
            pair = self._take(sites_a, sites_b, i, j, used) or self._take(
                sites_a, sites_b, j, i, used
            )
            if pair is None:
                raise ValueError(
                    f"topology edge ({i}, {j}) cannot be formed: neither residue "
                    "has a free site for the first reactant while the other has "
                    "one for the second. Check the monomer's site labels and "
                    "how many bonds each residue is asked to make."
                )
            occ_a, occ_b = pair
            used |= {*occ_a.values(), *occ_b.values()}
            yield {**occ_a, **occ_b}

    @staticmethod
    def _take(
        sites_a: dict[int, list[Binding]],
        sites_b: dict[int, list[Binding]],
        res_a: int,
        res_b: int,
        used: set[int],
    ) -> tuple[Binding, Binding] | None:
        """First free (first-reactant, second-reactant) site pair, or ``None``."""
        for occ_a in sites_a.get(res_a, ()):
            atoms_a = set(occ_a.values())
            if atoms_a & used:
                continue
            for occ_b in sites_b.get(res_b, ()):
                atoms_b = set(occ_b.values())
                if atoms_b & used or atoms_a & atoms_b:
                    continue
                return occ_a, occ_b
        return None

    @staticmethod
    def _index(
        context: MatchContext, component: int, map_number: int
    ) -> dict[int, list[Binding]]:
        """Residue id -> its occurrences of this reactant, ordered by site handle.

        A residue may legitimately carry several sites for the same reactant (a
        four-arm crosslinker carries four), so this is a list. The order is by
        bonding-atom handle, which makes the assembly deterministic.
        """
        out: dict[int, list[Binding]] = {}
        for occurrence in context.occurrences[component]:
            handle = occurrence[map_number]
            residue = context.world.get(handle, fields.RES_ID.key)
            if residue is None:
                raise ValueError(
                    f"atom {handle} matched a reactant site but carries no "
                    f"{fields.RES_ID.key}; expand the monomer library first"
                )
            out.setdefault(int(residue), []).append(occurrence)
        for occurrences in out.values():
            occurrences.sort(key=lambda occ: occ[map_number])
        return out
