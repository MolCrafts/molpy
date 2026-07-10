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
hydroxyl); charge is instead conserved by construction, by folding each cap's
charge onto its site atom on the *template* so a deleted atom carries exactly
zero. This typifier never writes a charge back.

It satisfies the region-typifier protocol (``scope`` + ``typify_region``) so a
:class:`~molpy.builder.assembly.GraphAssembler` retypes and patches each formed
junction back onto the network through its ``typifier=`` hook + ``RetypeCache``
— the patch-back is the assembler's job, not the wrapper's.

Unlike a typifier whose SMARTS patterns MolPy can read, AmberTools is a black box:
antechamber will not say how far it looks. Hence ``reach`` is a required argument
(``reach=2`` for GAFF, whose atom types are set by a one-to-two-bond environment).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from molpy.typifier.region import RegionTypes, TypeInfo
from molpy.typifier.scope import TypeScope
from molpy.core import fields

if TYPE_CHECKING:
    from molpy.builder.ambertools import AmberTools
    from molpy.typifier.affected_region import AffectedRegion


class AmberToolsTypifier:
    """Type an affected region's interior via AmberTools; accumulate its FF."""

    def __init__(self, amber: AmberTools, *, reach: int) -> None:
        """Bind an AmberTools wrapper and declare its receptive field.

        Args:
            amber: The antechamber/parmchk2/tleap wrapper.
            reach: Neighbourhood radius (bonds) that decides one GAFF atom type.
                **Required** — antechamber is a black box, so molpy cannot derive
                it the way it can for a compiled SMARTS pattern set. GAFF atom
                types are set by a 1–2 bond environment, so ``reach=2``; measure
                it for a bulky or fused junction rather than guessing.
        """
        self._amber = amber
        self._scope = TypeScope(reach=int(reach))
        # Merged force field of every distinct region typed so far (lazily set to
        # the first region's ff, then merged into). Holds the junction bonded
        # terms a linear chain lacks.
        self._forcefield: Any = None

    @property
    def scope(self) -> TypeScope:
        """The declared receptive field (see :meth:`__init__`)."""
        return self._scope

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
        """Interior atom **types**, keyed by canonical order.

        antechamber preserves atom order and the capped molecule keeps the region
        atoms first, so ``result``'s i-th atom is the region's i-th atom. Only the
        write-back set (``hops <= interior_reach``) is recorded; atoms further out
        exist to give it context and carry truncated context themselves. Charges
        are deliberately not captured — they are conserved locally by the
        reacter/crosslinker, not recomputed.

        Raises:
            ValueError: if an interior atom came back untyped — the extracted ball
                was too small for its context.
        """
        block = result.frame["atoms"]
        types = list(block[fields.TYPE.key])

        region_atoms = list(region.atoms)
        canon = region.canonical_order()
        pos_of_handle = {atom.handle: i for i, atom in enumerate(region_atoms)}
        canon_of_pos = {pos_of_handle[h]: idx for idx, h in enumerate(canon)}
        interior = {atom.handle for atom in region.interior}

        entries: list[tuple[int, TypeInfo]] = []
        for pos, atom in enumerate(region_atoms):
            if atom.handle not in interior:
                continue
            kind = str(types[pos])
            entries.append((canon_of_pos[pos], TypeInfo(type=kind or None, params=())))
        entries.sort(key=lambda entry: entry[0])

        untyped = [idx for idx, info in entries if info.type is None]
        if untyped:
            raise ValueError(
                "region interior atom(s) left untyped by antechamber at canonical "
                f"positions {untyped}: extract_radius={region.extract_radius} is "
                f"too small for interior_reach={region.interior_reach}"
            )
        return RegionTypes(
            atoms=tuple(entries), bonds=(), angles=(), dihedrals=(), impropers=()
        )
