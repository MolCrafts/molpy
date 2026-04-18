"""Morse bond potential (data-carrying; numerical kernel TBD)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, BondStyle, BondType

from .base import BondPotential


class BondMorse(BondPotential):
    name = "morse"
    type = "bond"


class BondMorseStyle(BondStyle):
    """Morse bond style. Parameters per type: ``D`` (well depth),
    ``alpha`` (steepness), ``r0`` (equilibrium length).
    """

    def __init__(self):
        super().__init__("morse")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        D: float,
        alpha: float,
        r0: float,
        name: str = "",
        **kwargs: Any,
    ) -> BondType:
        if not name:
            name = f"{itom.name}-{jtom.name}"
        bt = BondType(name, itom, jtom, D=D, alpha=alpha, r0=r0, **kwargs)
        self.types.add(bt)
        return bt
