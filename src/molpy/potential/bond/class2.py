"""Class2 (quartic) bond potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, BondStyle, BondType

from .base import BondPotential


class BondClass2(BondPotential):
    name = "class2"
    type = "bond"


class BondClass2Style(BondStyle):
    """Class2 quartic bond: ``E = k2*(r-r0)^2 + k3*(r-r0)^3 + k4*(r-r0)^4``."""

    def __init__(self):
        super().__init__("class2")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        r0: float,
        k2: float,
        k3: float,
        k4: float,
        name: str = "",
        **kwargs: Any,
    ) -> BondType:
        if not name:
            name = f"{itom.name}-{jtom.name}"
        bt = BondType(name, itom, jtom, r0=r0, k2=k2, k3=k3, k4=k4, **kwargs)
        self.types.add(bt)
        return bt
