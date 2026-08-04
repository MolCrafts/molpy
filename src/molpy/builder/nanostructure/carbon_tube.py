"""Carbon nanotube builder — thin facade over :class:`molrs.builder.CarbonTubeBuilder`."""

from __future__ import annotations

from math import isfinite

import molrs

from molpy.builder._finalize import Finalization, StructureFinalizer
from molpy.core.atomistic import Atomistic
from molpy.core.box import Box
from molpy.typifier.forcefield import ForceFieldParams


class CarbonTubeBuilder:
    """Exact single-wall carbon nanotube via molrs.

    Lattice, seam, coordinates, bonds, and cell are built in Rust. This class
    only validates constructor kwargs and applies MolPy finalization.
    """

    def __init__(
        self,
        n: int,
        m: int,
        *,
        length: float | None = None,
        cells: int | None = None,
        bond_length: float = 1.42,
        periodic: bool = False,
    ) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("m must be an integer")
        if n < 0 or m < 0:
            raise ValueError("n and m must be non-negative")
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be a bool")
        if length is not None and cells is not None:
            raise TypeError("length and cells are mutually exclusive")
        if cells is not None and (
            isinstance(cells, bool) or not isinstance(cells, int)
        ):
            raise TypeError("cells must be an integer")

        bond_length = float(bond_length)
        if not isfinite(bond_length) or bond_length <= 0.0:
            raise ValueError("bond_length must be finite and positive")

        self._native = molrs.builder.CarbonTubeBuilder(
            n,
            m,
            length=length,
            cells=cells,
            bond_length=bond_length,
            periodic=periodic,
        )
        self.n = n
        self.m = m
        self.cells = self._native.cells
        self.bond_length = bond_length
        self.periodic = periodic

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

    def cell(self, *, vacuum: float = 10.0) -> Box:
        """Return the molrs-generated simulation cell as a MolPy box."""
        vacuum = float(vacuum)
        if not isfinite(vacuum) or vacuum < 0.0:
            raise ValueError("vacuum must be finite and non-negative")
        return Box.from_box(self._native.cell(vacuum=vacuum))
