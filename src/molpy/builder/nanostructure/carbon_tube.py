"""Single-wall carbon nanotubes from an exact rolled-graphene topology."""

from __future__ import annotations

from functools import lru_cache
from math import ceil, cos, floor, gcd, pi, sin, sqrt

import numpy as np

from molpy.builder._finalize import Finalization, StructureFinalizer
from molpy.core.atomistic import Atomistic
from molpy.core.box import Box
from molpy.typifier.forcefield import ForceFieldParams

_SiteKey = tuple[int, int, int]
_CompiledTube = tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[int, int], ...],
    float,
    float,
]


class CarbonTubeBuilder:
    """Build a single-wall carbon nanotube with no public planning objects.

    The graphene quotient graph is compiled before an :class:`Atomistic` is
    mutated.  Integer lattice coordinates make the circumference and optional
    axial periodic boundary exact; bonds are inherited from graphene rather
    than guessed from rolled Cartesian distances.  Compiled immutable geometry
    and connectivity are cached across repeated builds.
    """

    def build(
        self,
        n: int,
        m: int,
        *,
        length: float | None = None,
        cells: int | None = None,
        bond_length: float = 1.42,
        periodic: bool = False,
        vacuum: float = 10.0,
        atom_type: str | None = None,
        charge: float = 0.0,
        finalize: Finalization | str = Finalization.ATOMS,
        bonded: ForceFieldParams | None = None,
    ) -> Atomistic:
        """Build a ``(n, m)`` nanotube and return a fresh molecular graph.

        Exactly one of ``length`` and ``cells`` may be supplied.  ``length``
        is rounded up to a whole translational unit; omitting both creates one
        unit (two for an image-safe periodic graph). ``periodic=True`` closes
        only the tube axis—the circumference is already closed by the nanotube
        topology. Higher-order topology is
        intentionally deferred by default and can be requested with
        ``finalize="topology"`` or ``"bonded"``.
        """
        n, m = self._validate_chirality(n, m)
        bond_length = self._positive_float("bond_length", bond_length)
        vacuum = self._nonnegative_float("vacuum", vacuum)
        charge = float(charge)
        if not np.isfinite(charge):
            raise ValueError("charge must be finite")
        if atom_type is not None and (not isinstance(atom_type, str) or not atom_type):
            raise ValueError("atom_type must be a non-empty string or None")
        if not isinstance(periodic, bool):
            raise TypeError("periodic must be a bool")

        axial_unit = self._axial_unit_length(n, m, bond_length)
        cell_count = self._resolve_cells(length, cells, axial_unit, periodic)
        coordinates, bond_indices, radius, axial_length = self._compile(
            n,
            m,
            cell_count,
            bond_length,
            periodic,
        )

        graph = Atomistic()
        atom_data = []
        for x, y, z in coordinates:
            attrs: dict[str, object] = {
                "element": "C",
                "charge": charge,
                "x": x,
                "y": y,
                "z": z,
            }
            if atom_type is not None:
                attrs["type"] = atom_type
            atom_data.append(attrs)
        atoms = graph.def_atoms(atom_data)
        graph.def_bonds([(atoms[i], atoms[j]) for i, j in bond_indices])

        diameter = 2.0 * radius
        transverse = diameter + 2.0 * vacuum
        graph["box"] = Box(
            np.diag([transverse, transverse, axial_length]),
            pbc=[False, False, periodic],
            origin=[-radius - vacuum, -radius - vacuum, 0.0],
        )
        return StructureFinalizer(Finalization(finalize), bonded).apply(graph)

    @staticmethod
    def _validate_chirality(n: int, m: int) -> tuple[int, int]:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n must be an integer")
        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError("m must be an integer")
        if n < 0 or m < 0 or (n == 0 and m == 0):
            raise ValueError("n and m must be non-negative and not both zero")
        if n * n + n * m + m * m < 4:
            raise ValueError(
                "chirality collapses distinct graphene neighbours; "
                "choose a larger nanotube"
            )
        return n, m

    @staticmethod
    def _positive_float(name: str, value: float) -> float:
        result = float(value)
        if not np.isfinite(result) or result <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return result

    @staticmethod
    def _nonnegative_float(name: str, value: float) -> float:
        result = float(value)
        if not np.isfinite(result) or result < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return result

    @staticmethod
    def _translation(n: int, m: int) -> tuple[int, int]:
        divisor = gcd(2 * m + n, 2 * n + m)
        return (2 * m + n) // divisor, -(2 * n + m) // divisor

    @classmethod
    def _axial_unit_length(cls, n: int, m: int, bond_length: float) -> float:
        t1, t2 = cls._translation(n, m)
        a1x = sqrt(3.0) * bond_length
        a2x = 0.5 * a1x
        a2y = 1.5 * bond_length
        return sqrt((t1 * a1x + t2 * a2x) ** 2 + (t2 * a2y) ** 2)

    @classmethod
    def _resolve_cells(
        cls,
        length: float | None,
        cells: int | None,
        axial_unit: float,
        periodic: bool,
    ) -> int:
        if length is not None and cells is not None:
            raise TypeError("length and cells are mutually exclusive")
        if cells is not None:
            if isinstance(cells, bool) or not isinstance(cells, int):
                raise TypeError("cells must be an integer")
            if cells <= 0:
                raise ValueError("cells must be positive")
            return cells
        if length is None:
            # A doubled axial cell also represents armchair tubes without
            # parallel periodic edges, which the molecular graph deliberately
            # does not model as a multigraph.
            return 2 if periodic else 1
        requested = cls._positive_float("length", length)
        return max(1, ceil(requested / axial_unit))

    @classmethod
    @lru_cache(maxsize=64)
    def _compile(
        cls,
        n: int,
        m: int,
        cells: int,
        bond_length: float,
        periodic: bool,
    ) -> _CompiledTube:
        """Compile immutable coordinates/connectivity before graph mutation."""
        t1, t2 = cls._translation(n, m)
        determinant = n * t2 - m * t1
        orientation = 1 if determinant > 0 else -1
        denominator = abs(3 * determinant)

        def projected(i: int, j: int, sublattice: int) -> tuple[int, int]:
            qx = 3 * i + sublattice
            qy = 3 * j + sublattice
            return (
                orientation * (qx * t2 - qy * t1),
                orientation * (n * qy - m * qx),
            )

        corners = ((0, 0), (n, m), (t1, t2), (n + t1, m + t2))
        i_min = floor(min(i for i, _ in corners)) - 2
        i_max = ceil(max(i for i, _ in corners)) + 2
        j_min = floor(min(j for _, j in corners)) - 2
        j_max = ceil(max(j for _, j in corners)) + 2

        unit_sites: dict[_SiteKey, tuple[int, int]] = {}
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for sublattice in (0, 1):
                    u_num, v_num = projected(i, j, sublattice)
                    if 0 <= u_num < denominator and 0 <= v_num < denominator:
                        unit_sites[(u_num, v_num, sublattice)] = (i, j)

        sites: dict[_SiteKey, tuple[int, int]] = {}
        for cell in range(cells):
            for (u_num, v_num, sublattice), (i, j) in unit_sites.items():
                sites[(u_num, v_num + cell * denominator, sublattice)] = (
                    i + cell * t1,
                    j + cell * t2,
                )

        expected_per_cell = 4 * (n * n + n * m + m * m) // gcd(2 * m + n, 2 * n + m)
        if len(sites) != expected_per_cell * cells:
            raise RuntimeError("internal nanotube lattice enumeration is incomplete")

        ordered_keys = sorted(sites, key=lambda key: (key[1], key[0], key[2]))
        indices = {key: index for index, key in enumerate(ordered_keys)}
        axial_denominator = cells * denominator

        def canonical_key(i: int, j: int, sublattice: int) -> _SiteKey | None:
            u_num, v_num = projected(i, j, sublattice)
            if periodic:
                v_num %= axial_denominator
            elif not 0 <= v_num < axial_denominator:
                return None
            return u_num % denominator, v_num, sublattice

        bonds: set[tuple[int, int]] = set()
        for key, (i, j) in sites.items():
            if key[2] != 0:
                continue
            for neighbour_i, neighbour_j in ((i, j), (i - 1, j), (i, j - 1)):
                neighbour = canonical_key(neighbour_i, neighbour_j, 1)
                if neighbour is None:
                    continue
                try:
                    pair = sorted((indices[key], indices[neighbour]))
                except KeyError as exc:  # pragma: no cover - invariant guard
                    raise RuntimeError(
                        "internal nanotube bond target is missing"
                    ) from exc
                bonds.add((pair[0], pair[1]))

        if periodic and len(bonds) != 3 * len(sites) // 2:
            raise ValueError(
                "the periodic cell is too short to represent distinct bonds; "
                "increase cells or length"
            )

        a1x = sqrt(3.0) * bond_length
        a2x = 0.5 * a1x
        a2y = 1.5 * bond_length
        circumference_x = n * a1x + m * a2x
        circumference_y = m * a2y
        circumference = sqrt(circumference_x**2 + circumference_y**2)
        radius = circumference / (2.0 * pi)
        axial_unit = cls._axial_unit_length(n, m, bond_length)

        coordinates = []
        for u_num, v_num, _ in ordered_keys:
            theta = 2.0 * pi * u_num / denominator
            coordinates.append(
                (
                    radius * cos(theta),
                    radius * sin(theta),
                    axial_unit * v_num / denominator,
                )
            )
        return tuple(coordinates), tuple(sorted(bonds)), radius, cells * axial_unit
