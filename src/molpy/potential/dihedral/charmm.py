"""CHARMM dihedral potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, DihedralStyle, DihedralType

from ..base import Potential


class DihedralCharmm(Potential):
    name = "charmm"
    type = "dihedral"


class DihedralCharmmStyle(DihedralStyle):
    """CHARMM proper: ``E = K*(1 + cos(n*phi - d))``, 1-4 weight ``w``."""

    def __init__(self):
        super().__init__("charmm")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        k: float,
        n: int,
        d: float,
        w: float = 0.0,
        name: str = "",
        **kwargs: Any,
    ) -> DihedralType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralType(name, itom, jtom, ktom, ltom, k=k, n=n, d=d, w=w, **kwargs)
        self.types.add(dt)
        return dt
