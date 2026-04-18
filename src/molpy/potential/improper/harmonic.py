"""Harmonic improper dihedral (data-carrying)."""

from __future__ import annotations

from typing import Any

from molpy.core.forcefield import AtomType, ImproperStyle, ImproperType

from ..base import Potential


class ImproperHarmonic(Potential):
    name = "harmonic"
    type = "improper"


class ImproperHarmonicStyle(ImproperStyle):
    """Harmonic improper: ``E = k * (chi - chi0)^2``."""

    def __init__(self):
        super().__init__("harmonic")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        k: float,
        chi0: float,
        name: str = "",
        **kwargs: Any,
    ) -> ImproperType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        it = ImproperType(name, itom, jtom, ktom, ltom, k=k, chi0=chi0, **kwargs)
        self.types.add(it)
        return it
