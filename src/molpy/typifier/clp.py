"""CL&P ionic-liquid force-field typifier.

CL&P stays in molpy (OPLS-AA moved to molrs). It is an OPLS-AA *overlay*: the
built-in force field is ``oplsaa.xml`` with ``clp.xml`` layered on top (layer 1),
so CL&P atom types (imidazolium ring, alkyl chain, and the BF4/PF6/NTf2/FSI/dca
anions) override the OPLS base while OPLS remains the fallback.

Atom typing itself is SMARTS-based and that matcher is owned by molrs, so the
only thing this typifier's ``match`` does is ask molrs for the atom types and
hand them to :class:`~molpy.typifier.forcefield.ForceFieldParams` — exactly the
"SMARTS owned by molrs, CL&P parameters stay a molpy overlay" split.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache
from typing import TYPE_CHECKING, override

from molrs.typifier import OPLSAATypifier

from molpy.core import fields
from molpy.core.atomistic import Atomistic
from molpy.typifier.base import Match, Typifier
from molpy.typifier.forcefield import ForceFieldParams

if TYPE_CHECKING:
    from collections.abc import Mapping

    from molpy.core.forcefield import ForceField
    from molpy.typifier.base import Annotation

# clp.xml sections the molrs OPLS *potential* reader understands. Its bonded
# sections use CL&P/foyer spellings (``<PeriodicTorsionForce>`` etc.) that the
# molrs OPLS reader rejects, and they are irrelevant to atom typing anyway — the
# bonded parameters are read from the overlay ForceField by ForceFieldParams.
_ATOMTYPE_SECTIONS = frozenset({"AtomTypes", "NonbondedForce"})


@lru_cache(maxsize=1)
def _clp_atomtypes_xml() -> str:
    """CL&P ``clp.xml`` reduced to the ``<AtomTypes>``/``<NonbondedForce>`` the
    molrs OPLS-AA typifier needs for SMARTS atom typing (bonded sections dropped)."""
    from molpy.data.forcefield import get_forcefield_path

    root = ET.parse(get_forcefield_path("clp.xml")).getroot()
    reduced = ET.Element(root.tag, root.attrib)
    for child in root:
        if child.tag in _ATOMTYPE_SECTIONS:
            reduced.append(child)
    return ET.tostring(reduced, encoding="unicode")


@lru_cache(maxsize=1)
def _clp_molrs_typifier() -> OPLSAATypifier:
    """Shared, stateless molrs SMARTS typifier over the CL&P overlay (compiling
    the SMARTS engine once instead of per :class:`ClpTypifier` construction)."""
    # strict=False so molrs's own bonded matching never errors — only the
    # atom-level type/class it assigns via the CL&P SMARTS defs is harvested.
    return OPLSAATypifier(_clp_atomtypes_xml(), strict=False)


class ClpTypifier(Typifier[Atomistic]):
    """CL&P ionic-liquid typifier — molrs SMARTS atom typing + molpy parameters.

    Args:
        forcefield: The CL&P-over-OPLS overlay; the built-in one by default.
        strict: Raise on an atom no SMARTS pattern matches, or a bonded term the
            force field does not parameterise.
    """

    def __init__(
        self, forcefield: ForceField | None = None, *, strict: bool = True
    ) -> None:
        self.ff = forcefield if forcefield is not None else self.load_forcefield()
        self._strict = strict
        self._params = ForceFieldParams(self.ff, strict=strict)
        self._smarts = _clp_molrs_typifier()

    @override
    def match(self, graph: Atomistic) -> Match:
        return self._params.match(graph, self._atom_types(graph))

    def _atom_types(self, graph: Atomistic) -> list[Mapping[str, Annotation]]:
        """Ask molrs which CL&P type (and class) each atom carries."""
        typed = self._smarts.typify(graph).to_frame()["atoms"]
        names = typed[fields.TYPE.key]
        classes = typed["class"] if "class" in typed else [None] * len(names)

        out: list[Mapping[str, Annotation]] = []
        for atom, name, class_name in zip(graph.atoms, names, classes, strict=True):
            if name in (None, ""):
                if self._strict:
                    raise ValueError(f"CL&P: no atom type matched for atom {atom}")
                out.append({})
                continue
            annotation: dict[str, Annotation] = {fields.TYPE.key: str(name)}
            if class_name not in (None, ""):
                annotation["class"] = str(class_name)
            out.append(annotation)
        return out

    @staticmethod
    def load_forcefield() -> ForceField:
        """Load the built-in CL&P force field as an OPLS-AA overlay."""
        from molpy.data.forcefield import get_forcefield_path
        from molpy.io.forcefield.xml import read_oplsaa_forcefield, read_xml_forcefield

        ff = read_oplsaa_forcefield("oplsaa.xml")
        return read_xml_forcefield(get_forcefield_path("clp.xml"), ff, layer=1)


__all__ = ["ClpTypifier"]
