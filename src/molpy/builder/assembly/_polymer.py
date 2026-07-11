"""Grow a polymer from CGSmiles over a monomer library.

``PolymerBuilder`` **is** a :class:`~molpy.builder.assembly._assembler.GraphAssembler`.
It owns the two things the kernel does not: a monomer library, and the CGSmiles
notation. That is why it is a class and not a helper that wires three objects
together — it carries data and it translates a notation into a world plus a
pairing rule. The internal IR (``CGSmilesIR.base_graph``) never reaches the user.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._assembler import GraphAssembler
from molpy.builder.assembly._library import MonomerLibrary
from molpy.builder.assembly._topology import TopologySelector
from molpy.core import fields
from molpy.parser.smiles import parse_cgsmiles

if TYPE_CHECKING:
    from molrs.fields import FieldSpec

    from molpy.builder.assembly._placer import Placer
    from molpy.core.atomistic import Atomistic
    from molpy.typifier.base import Typifier


class PolymerBuilder(GraphAssembler):
    """Stamp out repeat units and bond the adjacent ones.

    Example::

        eo = mp.read_smiles("OCCO")          # capped: ethylene glycol
        eo.atoms[0][fields.SITE] = "a"
        eo.atoms[3][fields.SITE] = "b"

        ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
        chain = PolymerBuilder(MonomerLibrary({"EO": eo}), ether).build("{[#EO]|1000}")
    """

    def __init__(
        self,
        library: MonomerLibrary | Mapping[str, Atomistic],
        reaction: molrs.Reaction,
        *,
        typifier: Typifier | None = None,
        reach: int | None = None,
        placer: Placer | None = None,
        label_field: FieldSpec = fields.SITE,
    ) -> None:
        super().__init__(
            reaction,
            typifier=typifier,
            reach=reach,
            placer=placer,
            label_field=label_field,
        )
        self._library = (
            library if isinstance(library, MonomerLibrary) else MonomerLibrary(library)
        )

    @property
    def library(self) -> MonomerLibrary:
        return self._library

    def build(self, cgsmiles: str) -> Atomistic:
        """Expand ``cgsmiles`` over the library and bond adjacent residues.

        ``build`` takes a *notation*, not a world, so it is not a second verb for
        the same operation: ``build == expand + assemble``. Reuse one builder
        across many chains and they share one retype cache, even at different
        lengths.
        """
        topology = parse_cgsmiles(cgsmiles).base_graph
        world = self._library.expand(topology)
        return self.assemble(world, TopologySelector(topology))
