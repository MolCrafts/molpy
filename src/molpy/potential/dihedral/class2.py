"""Class2 dihedral potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, DihedralStyle, DihedralType

from ..base import Potential


class DihedralClass2(Potential):
    name = "class2"
    type = "dihedral"


class DihedralClass2Style(DihedralStyle):
    """Class2 dihedral core term (three-term cosine expansion)."""

    def __init__(self):
        super().__init__("class2")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        k1: float,
        phi1: float,
        k2: float,
        phi2: float,
        k3: float,
        phi3: float,
        name: str = "",
        **kwargs: Any,
    ) -> DihedralType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralType(
            name,
            itom,
            jtom,
            ktom,
            ltom,
            k1=k1,
            phi1=phi1,
            k2=k2,
            phi2=phi2,
            k3=k3,
            phi3=phi3,
            **kwargs,
        )
        self.types.add(dt)
        return dt
