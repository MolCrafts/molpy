"""CVFF improper dihedral (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, ImproperStyle, ImproperType

from ..base import Potential


class ImproperCvff(Potential):
    name = "cvff"
    type = "improper"


class ImproperCvffStyle(ImproperStyle):
    """CVFF improper: ``E = k * (1 + d*cos(n*chi))``, ``d = +/-1``."""

    def __init__(self):
        super().__init__("cvff")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        k: float,
        d: int,
        n: int,
        name: str = "",
        **kwargs: Any,
    ) -> ImproperType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        it = ImproperType(name, itom, jtom, ktom, ltom, k=k, d=d, n=n, **kwargs)
        self.types.add(it)
        return it
