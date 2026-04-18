"""Morse pair potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, PairStyle, PairType

from ..base import Potential


class PairMorse(Potential):
    name = "morse"
    type = "pair"


class PairMorseStyle(PairStyle):
    """Morse: ``E = D0 * ((1 - exp(-alpha*(r-r0)))^2 - 1)``."""

    def __init__(self, **style_kwargs: Any):
        super().__init__("morse", **style_kwargs)

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        D0: float = 0.0,
        alpha: float = 0.0,
        r0: float = 0.0,
        name: str = "",
        **kwargs: Any,
    ) -> PairType:
        if jtom is None:
            jtom = itom
        if not name:
            name = f"{itom.name}-{jtom.name}"
        pt = PairType(name, itom, jtom, D0=D0, alpha=alpha, r0=r0, **kwargs)
        self.types.add(pt)
        return pt
