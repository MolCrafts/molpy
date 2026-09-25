"""Grow a polymer from a residue topology over a monomer library.

``PolymerBuilder`` **is** a :class:`~molpy.builder.assembly._assembler.GraphAssembler`.
It owns a monomer library and turns residue architecture into a world plus a
pairing rule. The **only** expand + apply entry is :meth:`build`; the
``build_*`` helpers only build a topology and call :meth:`build`.

Topology is a
:class:`~molpy.builder.assembly._residue_ir.ResidueTopology` built by
:mod:`~molpy.builder.assembly._residue_graph` constructors. SMILES for monomers
is ``SmilesIR`` via the rest of molpy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import molrs
from molpy.builder.assembly._assembler import GraphAssembler
from molpy.builder.assembly._residue_ir import ResidueTopology
from molpy.builder._finalize import Finalization
from molpy.builder.assembly._library import MonomerLibrary
from molpy.builder.assembly._residue_graph import (
    linear_topology,
    ring_topology,
    star_topology,
)
from molpy.core import fields

if TYPE_CHECKING:
    from molrs import Placer
    from molpy.core.atomistic import Atomistic
    from molpy.typifier.forcefield import ForceFieldParams


class PolymerBuilder(GraphAssembler):
    """Stamp out repeat units and bond the adjacent ones.

    **Sole assembly entry:** :meth:`build` (topology → expand → apply).

    **Shortcuts** (build topology, then call :meth:`build`):

    * :meth:`build_linear` → path of ``n`` identical residues
    * :meth:`build_sequence` → path from label list
    * :meth:`build_ring` → cycle
    * :meth:`build_star` → branched star

    Placement is opt-in: without ``placer`` every residue keeps its template
    coordinates, so the pasted copies sit on top of each other.

    Example::

        SiteMap(eo).label_elements("O", "a", "b")
        ether = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")
        builder = PolymerBuilder(
            MonomerLibrary({"EO": eo}), ether, placer=TracePlacer()
        )
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
        """Bind the monomer library and the assembly options.

        Args:
            library: Monomer templates by label, as a :class:`MonomerLibrary`
                or a plain mapping (wrapped in one). Every template must mark
                at least one atom with ``fields.SITE``.
            reaction: The reaction joining adjacent residues.
            typifier: Retypes each junction after the reaction batch; ``None``
                assigns no types.
            reach: Neighbourhood radius, in bonds, that decides one atom's
                type; required exactly when ``typifier`` is given.
            placer: Moves whole residues before the reaction so each forming
                bond starts at bonding range, e.g. ``TracePlacer()``. ``None``
                (default) means no placement: residues keep their template
                coordinates and stack on top of each other.
            label_field: Atom field holding the site labels.
            finalize: ``"atoms"``, ``"topology"`` (default) or ``"bonded"``.
            bonded: Force-field parameter assigner for ``finalize="bonded"``.

        Raises:
            TypeError: as :class:`GraphAssembler`.
            ValueError: as :class:`GraphAssembler`; also if the library is
                empty or a template marks no reaction site.
        """
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

    def build(self, topology: ResidueTopology) -> Atomistic:
        """Expand ``topology`` over the library and bond adjacent residues.

        This is the **only** path that expands the monomer library and runs
        :meth:`apply`. All ``build_*`` helpers end here. The topology is handed
        to :meth:`MonomerLibrary.expand` once; the pairing rule comes back with
        the world.

        Args:
            topology: Residue graph whose node labels are library keys, e.g.
                from ``linear_topology`` / ``ring_topology`` /
                ``star_topology``.

        Returns:
            The assembled polymer; each atom carries ``RES_ID`` (1-based
            residue position) and ``RES_NAME`` (monomer label).

        Raises:
            TypeError: if ``topology`` is not a :class:`ResidueTopology`
                (e.g. a notation string).
            ValueError: if the topology names a monomer the library lacks; if
                a topology edge cannot be formed because neither residue has a
                free site for the reaction's first reactant while the other
                has one for the second; or anything :meth:`apply` raises
                (placement failure, net-charge change).
        """
        if not isinstance(topology, ResidueTopology):
            raise TypeError(
                f"PolymerBuilder.build takes a ResidueTopology, got "
                f"{type(topology).__name__}; build one with linear_topology, "
                "ring_topology or star_topology"
            )
        expansion = self._library.expand(topology)
        return self.apply(expansion.world, expansion.pairing)

    def build_sequence(self, labels: Sequence[str]) -> Atomistic:
        """Linear path from library labels — shortcut for :meth:`build`.

        Args:
            labels: Monomer library keys in chain order.

        Returns:
            The polymer ``build(linear_topology(labels))`` returns.

        Raises:
            ValueError: if ``labels`` is empty, or as :meth:`build`.
        """
        return self.build(linear_topology(labels))

    def build_linear(self, label: str, n: int) -> Atomistic:
        """Homopolymer path of ``n`` residues — shortcut for :meth:`build`.

        Args:
            label: Monomer library key repeated ``n`` times.
            n: Number of residues.

        Returns:
            The polymer ``build(linear_topology([label] * n))`` returns.

        Raises:
            ValueError: if ``n < 1``, or as :meth:`build`.
        """
        if n < 1:
            raise ValueError(f"build_linear needs n >= 1, got {n}")
        return self.build(linear_topology([label] * n))

    def build_ring(self, label: str, n: int) -> Atomistic:
        """Macrocycle of ``n`` residues — shortcut for :meth:`build`.

        A placer forms the ring-closing bond but does not place it: the bond
        spans whatever distance placing the open chain left, so shorten it
        afterwards with a geometry optimization, or pass a placer with an
        explicit ring-shaped ``Trace``.

        Args:
            label: Monomer library key used for every residue.
            n: Number of residues.

        Returns:
            The polymer ``build(ring_topology(label, n))`` returns.

        Raises:
            ValueError: if ``n < 3``, or as :meth:`build`.
        """
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
        """Star polymer — shortcut for :meth:`build`.

        Args:
            core: Monomer library key of the central residue; its template
                needs at least ``n_arms`` free reaction sites.
            arm: Monomer library key repeated along every arm.
            n_arms: Number of arms.
            arm_length: Residues per arm, not counting the cap.
            cap: Optional monomer library key ending every arm.

        Returns:
            The polymer ``build(star_topology(...))`` returns.

        Raises:
            ValueError: if ``n_arms < 2`` or ``arm_length < 1``; if the core
                has too few free sites for ``n_arms`` (a bifunctional core has
                no site for a third arm); or as :meth:`build`.
        """
        return self.build(
            star_topology(core, arm, n_arms=n_arms, arm_length=arm_length, cap=cap)
        )
