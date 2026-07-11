"""Scoring a force-field bonded-term pattern against a concrete term.

A bonded type in an OPLS-style force field is keyed by a tuple of *component
names* (``("CT", "CT")``, ``("X", "CT", "CT", "X")``), and a component name may
be an atom type, an atom class, or the wildcard ``*``. Matching a concrete bond
or dihedral therefore means: does every component match the corresponding atom,
in either direction, and among all the types that match, which is the most
specific?

:class:`TypeClassIndex` answers both halves. It is read once off a
:class:`~molpy.core.forcefield.ForceField` and shared by every term matcher built
from that force field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from molpy.core.forcefield import AtomType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from molpy.core.forcefield import ForceField

#: Specificity of one matched pattern component. Only the ordering matters: a
#: type that resolves an endpoint exactly beats one that merely pins its class,
#: which beats a wildcard. Summed over the components of a term.
_SCORE_EXACT_TYPE = 3
_SCORE_CLASS = 1
_SCORE_WILDCARD = 0


class TypeClassIndex:
    """Resolve an atom type to its class, and a class to its overlay layer.

    Bond/angle/dihedral types store their class tuple only in the component
    *names*, and the :class:`~molpy.core.forcefield.AtomType` objects they point
    at carry no class. To match a bonded term we therefore resolve each atom's
    type to its class here and compare against those names.

    ``layer`` records the highest overlay layer of any atom type carrying a
    class, so a CL&P/CL&Pol class (layer >= 1) outranks a bare OPLS class
    (layer 0) when two parameter sets would otherwise tie.
    """

    def __init__(self, forcefield: ForceField) -> None:
        type_to_class: dict[str, str] = {}
        class_to_layer: dict[str, int] = {}
        for at in forcefield.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "*")
            cls = at.params.kwargs.get("class_", "*")
            layer = int(at.params.kwargs.get("layer") or 0)
            # Only real atom types (those an atom is actually assigned) define
            # the type->class map. Skip the wildcard class-placeholder atomtypes
            # the reader creates for class-keyed bond endpoints (type_="*",
            # name==class): their name can collide with a real type (e.g. CL&P
            # "FB" is both a BF4 fluorine type of class F AND the class of the
            # NTf2/FSI fluorines).
            if at_type != "*":
                type_to_class[at.name] = cls
            if cls and cls != "*":
                class_to_layer[cls] = max(class_to_layer.get(cls, 0), layer)
        self._type_to_class = type_to_class
        self._class_to_layer = class_to_layer

    def class_of(self, atom_type: str) -> str | None:
        """The class an atom type belongs to, or ``None`` if it declares none."""
        return self._type_to_class.get(atom_type)

    def layer_of(self, pattern_names: Iterable[str]) -> int:
        """The highest overlay layer any component of a bonded pattern sits on."""
        return max((self._class_to_layer.get(n, 0) for n in pattern_names), default=0)

    def score(
        self, pattern_names: Sequence[str], atom_types: Sequence[str]
    ) -> int | None:
        """Best specificity of a bonded pattern against an ordered atom list.

        Tries the term both forwards and reversed, because bonded terms are
        symmetric under end-for-end reversal.

        Returns:
            The summed specificity of the better-matching orientation, or
            ``None`` if neither orientation matches.
        """
        atoms = [(t, self.class_of(t)) for t in atom_types]
        best: int | None = None
        for order in (atoms, atoms[::-1]):
            total = 0
            for pname, (at_type, at_class) in zip(pattern_names, order, strict=True):
                component = self._component_score(pname, at_type, at_class)
                if component is None:
                    break
                total += component
            else:
                if best is None or total > best:
                    best = total
        return best

    @staticmethod
    def _component_score(
        pattern_name: str | None, atom_type: str, atom_class: str | None
    ) -> int | None:
        """Specificity of one pattern component against one atom, or ``None``."""
        if pattern_name is None or pattern_name == "*":
            return _SCORE_WILDCARD
        if pattern_name == atom_type:
            return _SCORE_EXACT_TYPE
        if atom_class is not None and pattern_name == atom_class:
            return _SCORE_CLASS
        return None
