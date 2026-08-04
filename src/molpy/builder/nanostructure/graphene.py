"""Graphene sheet builder — thin facade over :class:`molrs.builder.GrapheneBuilder`."""

from __future__ import annotations

from math import isfinite

import molrs

from molpy.builder._finalize import Finalization, StructureFinalizer
from molpy.core.atomistic import Atomistic
from molpy.core.box import Box
from molpy.typifier.forcefield import ForceFieldParams


class GrapheneBuilder:
    """Rectangular graphene (honeycomb) sheet via molrs.

    ``nx × ny`` honeycomb unit cells → ``2·nx·ny`` carbons. Bonds wrap in
    *xy* when ``periodic_xy`` is true (default).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        bond_length: float = 1.42,
        vacuum: float = 10.0,
        periodic_xy: bool = True,
    ) -> None:
        if isinstance(nx, bool) or not isinstance(nx, int) or nx <= 0:
            raise TypeError("nx must be a positive integer")
        if isinstance(ny, bool) or not isinstance(ny, int) or ny <= 0:
            raise TypeError("ny must be a positive integer")
        bond_length = float(bond_length)
        if not isfinite(bond_length) or bond_length <= 0.0:
            raise ValueError("bond_length must be finite and positive")
        vacuum = float(vacuum)
        if not isfinite(vacuum) or vacuum < 0.0:
            raise ValueError("vacuum must be finite and non-negative")
        if not isinstance(periodic_xy, bool):
            raise TypeError("periodic_xy must be a bool")

        self._native = molrs.builder.GrapheneBuilder(
            nx,
            ny,
            bond_length=bond_length,
            vacuum=vacuum,
            periodic_xy=periodic_xy,
        )
        self.nx = nx
        self.ny = ny
        self.bond_length = bond_length
        self.periodic_xy = periodic_xy

    def build(
        self,
        *,
        atom_type: str | None = None,
        charge: float = 0.0,
        finalize: Finalization | str = Finalization.ATOMS,
        bonded: ForceFieldParams | None = None,
    ) -> Atomistic:
        """Build a fresh molecular graph, optionally finalizing topology."""
        if atom_type is not None and (not isinstance(atom_type, str) or not atom_type):
            raise ValueError("atom_type must be a non-empty string or None")
        charge = float(charge)
        if not isfinite(charge):
            raise ValueError("charge must be finite")

        frame = self._native.build(atom_type=atom_type, charge=charge)
        graph = Atomistic.from_frame(frame)
        return StructureFinalizer(Finalization(finalize), bonded).apply(graph)

    def cell(self, *, vacuum: float | None = None) -> Box:
        """Return the molrs-generated simulation cell as a MolPy box."""
        if vacuum is None:
            return Box.from_box(self._native.cell())
        vacuum = float(vacuum)
        if not isfinite(vacuum) or vacuum < 0.0:
            raise ValueError("vacuum must be finite and non-negative")
        return Box.from_box(self._native.cell(vacuum=vacuum))
