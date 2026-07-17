"""GAFF typifier backed by AmberTools.

Decoupled from :class:`~molpy.builder.ambertools.AmberTools` (the raw
antechamber/parmchk2/tleap wrapper): this is the *typifier*. Its ``match`` drives
antechamber over the graph, reads back the GAFF atom types, and hands them to
:class:`~molpy.typifier.forcefield.ForceFieldParams` like every other force-field
typifier does.

**Types only, not charges.** Atom types + bonded params are charge-method
independent, so the graph is typed with ``gas`` charges — no ``sqm``/AM1-BCC
solve. Recomputing charges from a capped fragment would be both non-local and
biased (adding H to a cut ether-O makes antechamber see a hydroxyl); charge is
instead conserved by construction, by folding each cap's charge onto its site
atom on the *template* so a deleted atom carries exactly zero. This typifier
never writes a charge back.

Valence completion is not this class's business either: antechamber cannot
parameterise a sliced fragment, but whether a graph *is* a fragment is known to
whoever cut it. :meth:`~molpy.typifier.region.RegionTypes.of` completes every
region before typing it; a caller typing a whole molecule has nothing to complete.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.typifier.base import Match, Typifier
from molpy.typifier.forcefield import ForceFieldParams

if TYPE_CHECKING:
    from collections.abc import Mapping

    from molpy.builder.ambertools import AmberTools
    from molpy.typifier.base import Annotation


class AmberToolsTypifier(Typifier[Atomistic]):
    """Type a graph's atoms with GAFF via antechamber; accumulate its force field.

    Args:
        amber: The antechamber/parmchk2/tleap wrapper.
    """

    def __init__(self, amber: AmberTools) -> None:
        self._amber = amber
        # Merged force field of every distinct graph typed so far (lazily set to
        # the first result's ff, then merged into). Holds the junction bonded
        # terms a linear chain lacks.
        self._forcefield: Any = None

    @property
    def forcefield(self) -> Any:
        """The accumulated force field of everything typed so far (``None`` until one)."""
        return self._forcefield

    @override
    def match(self, graph: Atomistic) -> Match:
        """Run antechamber over ``graph`` and annotate it from the GAFF it yields."""
        result = self._amber.parameterize(
            graph,
            name=f"graph_{hash(graph) & 0xFFFFFFFF:08x}",
            charge_method="gas",
        )
        if self._forcefield is None:
            self._forcefield = result.ff
        else:
            self._forcefield.merge(result.ff)

        # antechamber preserves atom order, so its i-th atom is the graph's i-th.
        # strict=False: a cap sits at the edge of a sliced fragment and may carry
        # a bonded term GAFF does not parameterise; the region's interior guard,
        # not this matcher, is what forbids an undecided type where it matters.
        params = ForceFieldParams(result.ff, strict=False)
        return params.match(graph, self._atom_types(result))

    @staticmethod
    def _atom_types(result: Any) -> list[Mapping[str, Annotation]]:
        """The GAFF type antechamber gave each atom. Charges are not harvested."""
        block = result.frame["atoms"]
        return [
            {fields.TYPE.key: str(name)} if name not in (None, "") else {}
            for name in block[fields.TYPE.key]
        ]
