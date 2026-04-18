"""Class2 angle potential + cross-terms (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AngleStyle, AngleType, AtomType

from ..base import Potential


class AngleClass2(Potential):
    name = "class2"
    type = "angle"


class AngleClass2BondBond(Potential):
    name = "class2/bb"
    type = "angle"


class AngleClass2BondAngle(Potential):
    name = "class2/ba"
    type = "angle"


class AngleClass2Style(AngleStyle):
    """Class2 angle core term: ``E = k2*(th-th0)^2 + k3*(th-th0)^3 + k4*(th-th0)^4``."""

    def __init__(self):
        super().__init__("class2")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        theta0: float,
        k2: float,
        k3: float,
        k4: float,
        name: str = "",
        **kwargs: Any,
    ) -> AngleType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}"
        at = AngleType(
            name, itom, jtom, ktom, theta0=theta0, k2=k2, k3=k3, k4=k4, **kwargs
        )
        self.types.add(at)
        return at


class AngleClass2BondBondStyle(AngleStyle):
    """Bond-Bond cross-term for class2 angle."""

    def __init__(self):
        super().__init__("class2/bb")


class AngleClass2BondAngleStyle(AngleStyle):
    """Bond-Angle cross-term for class2 angle."""

    def __init__(self):
        super().__init__("class2/ba")
