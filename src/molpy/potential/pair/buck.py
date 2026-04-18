"""Buckingham pair potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, PairStyle, PairType

from ..base import Potential


class PairBuck(Potential):
    name = "buck"
    type = "pair"


class PairBuckStyle(PairStyle):
    """Buckingham: ``E = A*exp(-r/rho) - C/r^6``."""

    def __init__(self, **style_kwargs: Any):
        super().__init__("buck", **style_kwargs)

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        A: float = 0.0,
        rho: float = 1.0,
        C: float = 0.0,
        name: str = "",
        **kwargs: Any,
    ) -> PairType:
        if jtom is None:
            jtom = itom
        if not name:
            name = f"{itom.name}-{jtom.name}"
        pt = PairType(name, itom, jtom, A=A, rho=rho, C=C, **kwargs)
        self.types.add(pt)
        return pt
