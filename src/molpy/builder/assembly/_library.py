"""Monomer templates, and the world they expand into.

A repeat unit is an ordinary capped molecule with a few of its atoms named
through :data:`~molpy.core.fields.SITE`. Expanding a topology stamps each pasted
copy with ``RES_ID`` and ``RES_NAME`` — a repeat unit *is* a residue, and that
identity is what a PDB or a prmtop wants downstream. It is not a build-time
marker to be scrubbed afterwards.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from molpy.core import fields
from molpy.builder.assembly._topology import TopologySelector
from molpy.core.atomistic import Atomistic

if TYPE_CHECKING:
    from molpy.parser.smiles.cgsmiles_ir import CGSmilesGraphIR


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
        self._templates = dict(templates)

    def __contains__(self, label: object) -> bool:
        return label in self._templates

    def __getitem__(self, label: str) -> Atomistic:
        return self._templates[label]

    def expand(self, topology: CGSmilesGraphIR) -> Atomistic:
        """Paste one copy of each topology node's template into a fresh world.

        Each copy carries ``RES_ID`` (the node id) and ``RES_NAME`` (the monomer
        label). No geometry, no reaction: ``O(sum of template sizes)``.

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
        return world
