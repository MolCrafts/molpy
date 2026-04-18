"""Multi-harmonic dihedral potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, DihedralStyle, DihedralType

from ..base import Potential


class DihedralMultiHarmonic(Potential):
    name = "multi/harmonic"
    type = "dihedral"


class DihedralMultiHarmonicStyle(DihedralStyle):
    """Multi-harmonic: ``E = sum_{n=1..5} A_n * cos^(n-1)(phi)``."""

    def __init__(self):
        super().__init__("multi/harmonic")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        a1: float,
        a2: float,
        a3: float,
        a4: float,
        a5: float,
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
            a1=a1,
            a2=a2,
            a3=a3,
            a4=a4,
            a5=a5,
            **kwargs,
        )
        self.types.add(dt)
        return dt
