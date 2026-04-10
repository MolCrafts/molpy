"""Base class for all typifiers."""

from abc import ABC, abstractmethod
from typing import Any

from molpy.core.forcefield import AtomType, ForceField


class TypifierBase(ABC):
    """Base class for all typifiers."""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        self.ff = forcefield
        self.strict = strict

    @abstractmethod
    def typify(self, elem: Any) -> Any:
        """Assign type to element."""


def atomtype_matches(atomtype: AtomType, type_str: str) -> bool:
    """Check whether an atom type matches a target type/class string.

    Match rules:
    - ``X`` or ``*`` in ``type_``/``class_`` acts as a wildcard
    - otherwise exact match by ``type_`` or ``class_``
    """
    at_type = atomtype.params.kwargs.get("type_", "X")
    at_class = atomtype.params.kwargs.get("class_", "X")

    if at_type in {"X", "*"} or at_class in {"X", "*"}:
        return True

    return at_type == type_str or at_class == type_str
