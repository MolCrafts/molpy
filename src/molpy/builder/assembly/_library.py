"""Monomer templates, and the world they expand into.

A repeat unit is an ordinary capped molecule with a few of its atoms named
through :data:`~molpy.core.fields.SITE`. Expanding a topology stamps each pasted
copy with ``RES_ID`` and ``RES_NAME`` — a repeat unit *is* a residue, and that
identity is what a PDB or a prmtop wants downstream. It is not a build-time
marker to be scrubbed afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from molpy.core import fields
from molpy.builder.assembly._topology import TopologySelector
from molpy.core.atomistic import Atomistic

if TYPE_CHECKING:
    from molpy.builder.assembly._residue_ir import ResidueTopology


@dataclass(frozen=True)
class Expansion:
    """One :meth:`MonomerLibrary.expand` result: pasted world + pairing rule.

    The topology enters :meth:`~MonomerLibrary.expand` once and leaves as this
    pair, so the same ``ResidueTopology`` is never handed to a second object.
    """

    world: Atomistic
    pairing: TopologySelector


class MonomerLibrary:
    """Named monomer templates, validated once and pasted on demand."""

    def __init__(self, templates: Mapping[str, Atomistic]) -> None:
        """Bind and validate the templates.

        Raises:
            ValueError: if a template names no reaction site. A monomer with no
                ``SITE`` atom can never bond to anything, which is a modelling
                error and not a graph MolPy should silently assemble.
        """
        if not templates:
            raise ValueError("monomer library is empty")
        for label, template in templates.items():
            if not any(atom.get(fields.SITE) for atom in template.atoms):
                raise ValueError(
                    f"monomer {label!r} marks no reaction site: set "
                    f"atom[fields.SITE] on the atoms that may react"
                )
        # A builder's compiled local-environment cache is only valid for the
        # templates it validated. Keep private snapshots so later mutation of a
        # caller-owned graph (or of a graph returned by ``__getitem__``) cannot
        # silently stale that cache.
        self._templates = {
            label: template.copy() for label, template in templates.items()
        }

    def __contains__(self, label: object) -> bool:
        return label in self._templates

    def __getitem__(self, label: str) -> Atomistic:
        """Return an independent copy of the named template."""
        return self._templates[label].copy()

    def expand(self, topology: ResidueTopology) -> Expansion:
        """Paste one copy of each topology node's template into a fresh world.

        Each copy carries ``RES_ID`` — the node's 1-based position in
        ``topology.nodes`` (not its ``ResidueNode.id``, which is a
        process-wide counter) — and ``RES_NAME`` (the monomer label). No
        geometry, no reaction: every copy keeps its template coordinates, and
        the cost is ``O(sum of template sizes)``.

        Args:
            topology: Residue graph whose node labels are keys of this library.

        Returns:
            The pasted world together with the pairing rule derived from the
            same topology, so the topology is not handed out a second time.

        Raises:
            ValueError: if the topology names a monomer the library lacks.
        """
        missing = {node.label for node in topology.nodes} - set(self._templates)
        if missing:
            raise ValueError(
                f"topology names monomer(s) {sorted(missing)} that the library "
                f"lacks; it has {sorted(self._templates)}"
            )

        residue_of = TopologySelector.residue_ids(topology)
        world = Atomistic()
        for node in topology.nodes:
            copy = self._templates[node.label].copy()
            for atom in copy.atoms:
                atom[fields.RES_ID] = residue_of[node.id]
                atom[fields.RES_NAME] = node.label
            world.merge(copy)
        return Expansion(world=world, pairing=TopologySelector(topology))
