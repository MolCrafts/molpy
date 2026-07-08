"""GAFF region typifier backed by AmberTools.

Decoupled from :class:`~molpy.builder.ambertools.AmberTools` (the raw
antechamber/parmchk2/tleap wrapper): this is the *typifier*. It drives AmberTools
to parameterise an affected region and reads back a
:class:`~molpy.typifier.region.RegionTypes` snapshot of the interior atom
**types**, while accumulating the region's bonded force-field terms (the junction
angles/dihedrals a linear chain lacks).

**Types only, not charges.** Atom types + bonded params are charge-method
independent, so the region is typed with ``gas`` charges — no per-fragment
``sqm``/AM1-BCC solve. Recomputing junction charges from a capped fragment would
be both non-local and biased (capping a cut ether-O makes antechamber see a
hydroxyl); charge is instead conserved locally by the reacter/crosslinker folding
each leaving group's charge onto its anchor atom. This typifier never writes a
charge back.

It satisfies the region-typifier protocol (``context_radius`` + ``typify_region``)
so a :class:`~molpy.builder.crosslink.Crosslinker` (or reacter) retypes and
patches each formed junction back onto the network through its ``typifier=`` hook
+ ``RetypeCache`` — the patch-back is the reacter's job, not the wrapper's.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molpy.typifier.region import RegionTypes, TypeInfo

if TYPE_CHECKING:
    from molpy.builder.ambertools import AmberTools
    from molpy.core.affected_region import AffectedRegion

#: Default extraction radius (bonds). GAFF atom types are set by a 1–2 bond
#: environment, so a small ball types the junction interior identically to a
#: large one while keeping the retype cache from fragmenting on far-field edits.
_DEFAULT_CONTEXT_RADIUS = 2


class AmberToolsTypifier:
    """Type an affected region's interior via AmberTools; accumulate its FF."""

    def __init__(
        self, amber: AmberTools, *, context_radius: int = _DEFAULT_CONTEXT_RADIUS
    ) -> None:
        self._amber = amber
        self._context_radius = int(context_radius)
        # Merged force field of every distinct region typed so far (lazily set to
        # the first region's ff, then merged into). Holds the junction bonded
        # terms a linear chain lacks.
        self._forcefield: Any = None

    @property
    def context_radius(self) -> int:
        """Extraction ball depth (bonds) for a retyped region — user-tunable.

        GAFF atom typing reaches only 1–2 bonds, so the default (2) types the
        junction the same as a wider ball would. Raise it for a bulky or fused
        junction whose interior atoms need more surrounding context; a smaller
        radius also keeps the retype cache from splitting on unrelated nearby
        edits (see :func:`molpy.core.region_radius`).
        """
        return self._context_radius

    @property
    def forcefield(self) -> Any:
        """The accumulated force field of all typed regions (``None`` until one)."""
        return self._forcefield

    def typify_region(self, region: AffectedRegion) -> RegionTypes:
        """Parameterise ``region`` (capped to a valid molecule) and snapshot its
        interior GAFF **types**; accumulate its bonded parameters.

        Typed with ``gas`` charges: only atom types + bonded params are read back,
        both charge-method independent, so the ``sqm`` charge solve is skipped.
        ``complete_valence`` caps the sliced fragment to a valid molecule for
        antechamber; the recomputed types are snapshotted, charges are not.
        """
        result = self._amber.parameterize(
            region.complete_valence(),
            name=f"region_{hash(region) & 0xFFFFFFFF:08x}",
            charge_method="gas",
        )
        if self._forcefield is None:
            self._forcefield = result.ff
        else:
            self._forcefield.merge(result.ff)
        return self._snapshot(region, result)

    @staticmethod
    def _snapshot(region: AffectedRegion, result: Any) -> RegionTypes:
        """Interior (non-boundary) atom **types**, keyed by canonical order.

        antechamber preserves atom order and the capped molecule keeps the region
        atoms first, so ``result``'s i-th atom is the region's i-th atom. Boundary
        atoms carry truncated context and are dropped (same contract as
        :func:`~molpy.typifier.region.typify_region`). Charges are deliberately
        not captured — they are conserved locally by the reacter, not recomputed.
        """
        block = result.frame["atoms"]
        types = list(block["type"])

        region_atoms = list(region.atoms)
        canon = region.canonical_order()
        pos_of_handle = {atom.handle: i for i, atom in enumerate(region_atoms)}
        canon_of_pos = {pos_of_handle[h]: idx for idx, h in enumerate(canon)}
        boundary = {atom.handle for atom in region.boundary}

        entries: list[tuple[int, TypeInfo]] = []
        for pos, atom in enumerate(region_atoms):
            if atom.handle in boundary:
                continue
            entries.append(
                (canon_of_pos[pos], TypeInfo(type=str(types[pos]), params=()))
            )
        entries.sort(key=lambda entry: entry[0])
        return RegionTypes(atoms=tuple(entries), bonds=(), angles=(), dihedrals=())
