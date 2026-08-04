"""Grow a polymer from a residue topology over a monomer library.

``PolymerBuilder`` **is** a :class:`~molpy.builder.assembly._assembler.GraphAssembler`.
It owns a monomer library and turns residue architecture into a world plus a
pairing rule. The **only** expand + assemble entry is :meth:`build`; the
``build_*`` helpers only build a topology and call :meth:`build`.

Topology is a
:class:`~molpy.builder.assembly._cgsmiles_ir.CGSmilesGraphIR` built by
:mod:`~molpy.builder.assembly._residue_graph` constructors. SMILES for monomers
is :class:`molrs.io.SmilesIR` via the rest of molpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._assembler import GraphAssembler
from molpy.builder.assembly._cgsmiles_ir import CGSmilesGraphIR
from molpy.builder.assembly._finalize import Finalization
from molpy.builder.assembly._library import MonomerLibrary
from molpy.builder.assembly._residue_graph import (
    linear_topology,
    ring_topology,
    star_topology,
)
from molpy.builder.assembly._topology import TopologySelector
from molpy.core import fields

if TYPE_CHECKING:
    from molpy.builder.assembly._placer import Placer
    from molpy.core.atomistic import Atomistic
    from molpy.typifier.forcefield import ForceFieldParams


class PolymerBuilder(GraphAssembler):
    """Stamp out repeat units and bond the adjacent ones.

    **Sole assembly entry:** :meth:`build` (topology → expand → assemble).

    **Shortcuts** (build topology, then call :meth:`build`):

    * :meth:`build_linear` → path of ``n`` identical residues
    * :meth:`build_sequence` → path from label list
    * :meth:`build_ring` → cycle
    * :meth:`build_star` → branched star

    Example::

        SiteMap(eo).label_elements("O", "a", "b")
        ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
        builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether)
        chain = builder.build_linear("EO", 20)
    """

    def __init__(
        self,
        library: MonomerLibrary | Mapping[str, Atomistic],
        reaction: molrs.Reaction,
        *,
        typifier: molrs.ff.Typifier | None = None,
        reach: int | None = None,
        placer: Placer | None = None,
        label_field: str = fields.SITE,
        finalize: Finalization | str = Finalization.TOPOLOGY,
        bonded: ForceFieldParams | None = None,
    ) -> None:
        super().__init__(
            reaction,
            typifier=typifier,
            reach=reach,
            placer=placer,
            label_field=label_field,
            finalize=finalize,
            bonded=bonded,
        )
        self._library = (
            library if isinstance(library, MonomerLibrary) else MonomerLibrary(library)
        )

    @property
    def library(self) -> MonomerLibrary:
        return self._library

    def build(self, topology: CGSmilesGraphIR) -> Atomistic:
        """Expand ``topology`` over the library and bond adjacent residues.

        This is the **only** path that expands the monomer library and runs
        :meth:`assemble`. All ``build_*`` helpers end here.
        """
        world = self._library.expand(topology)
        return self.assemble(world, TopologySelector(topology))

    def build_sequence(self, labels: Sequence[str]) -> Atomistic:
        """Linear path from library labels — shortcut for :meth:`build`."""
        return self.build(linear_topology(labels))

    def build_linear(self, label: str, n: int) -> Atomistic:
        """Homopolymer path of ``n`` residues — shortcut for :meth:`build`.

        Raises:
            ValueError: if ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"build_linear needs n >= 1, got {n}")
        return self.build(linear_topology([label] * n))

    def build_ring(self, label: str, n: int) -> Atomistic:
        """Macrocycle of ``n`` residues (``n >= 3``) — shortcut for :meth:`build`."""
        return self.build(ring_topology(label, n))

    def build_star(
        self,
        core: str,
        arm: str,
        *,
        n_arms: int,
        arm_length: int,
        cap: str | None = None,
    ) -> Atomistic:
        """Star polymer — shortcut for :meth:`build`."""
        return self.build(
            star_topology(core, arm, n_arms=n_arms, arm_length=arm_length, cap=cap)
        )
