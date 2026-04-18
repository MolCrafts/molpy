"""Class2 improper dihedral (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, ImproperStyle, ImproperType

from ..base import Potential


class ImproperClass2(Potential):
    name = "class2"
    type = "improper"


class ImproperClass2Style(ImproperStyle):
    """Class2 improper (out-of-plane wag)."""

    def __init__(self):
        super().__init__("class2")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        k: float,
        chi0: float,
        name: str = "",
        **kwargs: Any,
    ) -> ImproperType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        it = ImproperType(name, itom, jtom, ktom, ltom, k=k, chi0=chi0, **kwargs)
        self.types.add(it)
        return it
