"""Class2 (12-9) LJ pair potential (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, PairStyle, PairType

from ..base import Potential


class PairLJClass2(Potential):
    name = "lj/class2"
    type = "pair"


class PairLJClass2Style(PairStyle):
    """Class2 LJ (12-9): ``E = epsilon * (2*(sigma/r)^9 - 3*(sigma/r)^6)``."""

    def __init__(self, **style_kwargs: Any):
        super().__init__("lj/class2", **style_kwargs)

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        epsilon: float = 0.0,
        sigma: float = 0.0,
        name: str = "",
        **kwargs: Any,
    ) -> PairType:
        if jtom is None:
            jtom = itom
        if not name:
            name = f"{itom.name}-{jtom.name}"
        pt = PairType(name, itom, jtom, epsilon=epsilon, sigma=sigma, **kwargs)
        self.types.add(pt)
        return pt
