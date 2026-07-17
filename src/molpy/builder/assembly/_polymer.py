"""Grow a polymer from CGSmiles over a monomer library.

``PolymerBuilder`` **is** a :class:`~molpy.builder.assembly._assembler.GraphAssembler`.
It owns a monomer library and turns residue architecture into a world plus a
pairing rule. The **only** expand + assemble entry is :meth:`build`; the
``build_*`` helpers only format CGSmiles and call :meth:`build`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._assembler import GraphAssembler
from molpy.builder.assembly._finalize import Finalization
from molpy.builder.assembly._library import MonomerLibrary
from molpy.builder.assembly._residue_graph import (
    linear_cgsmiles,
    ring_cgsmiles,
    star_cgsmiles,
)
from molpy.builder.assembly._topology import TopologySelector
from molpy.core import fields
from molpy.parser.smiles import parse_cgsmiles

if TYPE_CHECKING:
    from molrs.fields import FieldSpec

    from molpy.builder.assembly._placer import Placer
    from molpy.core.atomistic import Atomistic
    from molpy.typifier.forcefield import ForceFieldParams


class PolymerBuilder(GraphAssembler):
    """Stamp out repeat units and bond the adjacent ones.

    **Sole assembly entry:** :meth:`build` (CGSmiles → expand → assemble).

    **Shortcuts** (format notation, then call :meth:`build`):

    * :meth:`build_linear` → ``{[#EO]|n}``
    * :meth:`build_sequence` → ``{[#A][#B]…}``
    * :meth:`build_ring` → ring digits
    * :meth:`build_star` → branched CGSmiles

    Example::

        SiteMap(eo).label_elements("O", "a", "b")
        ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
        builder = PolymerBuilder(MonomerLibrary({"EO": eo}), ether)
        chain = builder.build_linear("EO", 20)   # == build("{[#EO]|20}")
    """

    def __init__(
        self,
        library: MonomerLibrary | Mapping[str, Atomistic],
        reaction: molrs.Reaction,
        *,
        typifier: molrs.Typifier | None = None,
        reach: int | None = None,
        placer: Placer | None = None,
        label_field: FieldSpec = fields.SITE,
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

    def build(self, cgsmiles: str) -> Atomistic:
        """Expand ``cgsmiles`` over the library and bond adjacent residues.

        This is the **only** path that expands the monomer library and runs
        :meth:`assemble`. All ``build_*`` helpers end here.

        ``build`` takes a *notation*, not a world: ``build == expand + assemble``.
        Reuse one builder across many chains to share the compiled local-product
        cache.
        """
        topology = parse_cgsmiles(cgsmiles).base_graph
        world = self._library.expand(topology)
        return self.assemble(world, TopologySelector(topology))

    def build_sequence(self, labels: Sequence[str]) -> Atomistic:
        """Linear path from library labels — shortcut for :meth:`build`.

        Example::

            builder.build_sequence(["EO"] * 20)       # == build("{[#EO]|20}")
            builder.build_sequence(["A"] * 8 + ["B"] * 4)
        """
        return self.build(linear_cgsmiles(labels))

    def build_linear(self, label: str, n: int) -> Atomistic:
        """Homopolymer path of ``n`` residues — shortcut for :meth:`build`.

        ``build_linear("EO", 5)`` is exactly ``build("{[#EO]|5}")``.

        Raises:
            ValueError: if ``n < 1``.
        """
        if n < 1:
            raise ValueError(f"build_linear needs n >= 1, got {n}")
        return self.build(linear_cgsmiles([label] * n))

    def build_ring(self, label: str, n: int) -> Atomistic:
        """Macrocycle of ``n`` residues (``n >= 3``) — shortcut for :meth:`build`."""
        return self.build(ring_cgsmiles(label, n))

    def build_star(
        self,
        core: str,
        arm: str,
        *,
        n_arms: int,
        arm_length: int,
        cap: str | None = None,
    ) -> Atomistic:
        """Star polymer — shortcut for :meth:`build` with branched CGSmiles.

        ``core`` must offer at least ``n_arms`` free sites for the reaction.
        Optional ``cap`` is a monofunctional end residue on every arm.
        """
        return self.build(
            star_cgsmiles(
                core,
                arm,
                n_arms=n_arms,
                arm_length=arm_length,
                cap=cap,
            )
        )
