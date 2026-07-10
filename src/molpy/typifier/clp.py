"""CL&P ionic-liquid force-field typifier.

CL&P stays in molpy (OPLS-AA moved to molrs). It is an OPLS-AA *overlay*: the
built-in force field is ``oplsaa.xml`` with ``clp.xml`` layered on top (layer 1),
so CL&P atom types (imidazolium ring, alkyl chain, and the BF4/PF6/NTf2/FSI/dca
anions) override the OPLS base while OPLS remains the fallback.

Atom typing itself is SMARTS-based and that matcher is owned by molrs, so this
typifier delegates atom typing to molrs's :class:`~molrs.typifier.OPLSAATypifier`
(fed only the CL&P ``<AtomTypes>``/``<NonbondedForce>`` overlay) and keeps the
molpy-side pair/bond/angle/dihedral typifiers for the parameters — exactly the
"OPLS/SMARTS owned by molrs, CL&P stays a molpy overlay" split.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import lru_cache

from molrs.typifier import OPLSAATypifier

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import ForceField
from molpy.core import fields

from .atomistic import ForceFieldTypifier

# clp.xml sections the molrs OPLS *potential* reader understands. Its bonded
# sections use CL&P/foyer spellings (``<PeriodicTorsionForce>`` etc.) that the
# molrs OPLS reader rejects, and they are irrelevant to atom typing anyway — the
# molpy-side bond/angle/dihedral typifiers read them from the overlay ForceField.
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
    return OPLSAATypifier(_clp_atomtypes_xml(), strict=False)


class _ClpAtomTypifier:
    """Assign CL&P atom ``type``/``class`` by delegating to molrs's SMARTS-based
    OPLS-AA typifier, then writing the result back onto a molpy structure."""

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict
        # strict=False so molrs's own bonded matching never errors — we harvest
        # only the atom-level type/class it assigns via the CL&P SMARTS defs.
        self._typifier = _clp_molrs_typifier()

    def typify(self, struct: Atomistic) -> Atomistic:
        """Return a copy of ``struct`` with CL&P ``type``/``class`` on every atom."""
        typed_atoms = self._typifier.typify(struct).to_frame()["atoms"]
        types = typed_atoms[fields.TYPE.key]
        classes = (
            typed_atoms["class"] if "class" in typed_atoms else [None] * len(types)
        )

        new_struct = struct.copy()
        for atom, type_name, class_name in zip(new_struct.atoms, types, classes):
            resolved = None if type_name in (None, "") else str(type_name)
            if resolved is None:
                if self.strict:
                    raise ValueError(f"CL&P: no atom type matched for atom {atom}")
                continue
            extra = {} if class_name in (None, "") else {"class": str(class_name)}
            atom.update(type=resolved, **extra)
        return new_struct


class ClpTypifier(ForceFieldTypifier):
    """CL&P ionic-liquid typifier — molrs SMARTS atom typing + molpy parameters."""

    def __init__(self, forcefield: ForceField | None = None, **kwargs: object) -> None:
        if forcefield is None:
            forcefield = self.load_forcefield()
        super().__init__(forcefield, **kwargs)

    def _init_typifiers(self) -> None:
        if not self.skip_atom_typing:
            self.atom_typifier = _ClpAtomTypifier(strict=self.strict_typing)
        super()._init_typifiers()

    @staticmethod
    def load_forcefield() -> ForceField:
        """Load the built-in CL&P force field as an OPLS-AA overlay."""
        from molpy.data.forcefield import get_forcefield_path
        from molpy.io.forcefield.xml import read_oplsaa_forcefield, read_xml_forcefield

        ff = read_oplsaa_forcefield("oplsaa.xml")
        return read_xml_forcefield(get_forcefield_path("clp.xml"), ff, layer=1)


__all__ = ["ClpTypifier"]
