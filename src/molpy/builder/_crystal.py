"""Crystal lattice builder.

Tile a Bravais lattice over a range of unit cells and (optionally) clip the
result to a geometric region (any native region: ``mp.core.Cuboid``,
``mp.core.Sphere``, their ``&`` / ``|`` / ``~`` compositions). The unit cell is a
:class:`~molpy.Box`, which owns fractional ↔ Cartesian conversion; molrs has
no crystal builder, so the lattice and its space groups are molpy's.

Example:
    >>> import molpy as mp
    >>> lat = mp.builder.Lattice.fcc(a=3.52, species="Ni")
    >>> structure = lat.build(repeats=(4, 4, 4))
    >>> # or clip a 30 Å cube out of a larger tile:
    >>> structure = lat.build(mp.core.Cuboid.cube(30.0))
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from molrs.core import (
    Atomistic,
    Box,
    Cuboid,
    Cylinder,
    Ellipsoid,
    Frame,
    HalfSpace,
    Parallelepiped,
    Polyhedron,
    Region,
    Sphere,
    SphereUnion,
)
from numpy.typing import ArrayLike

from molpy.builder._symmetry import SpaceGroup

__all__ = ["Lattice", "Site"]

#: Fractional slack on the inferred tile, so an atom on the region's
#: boundary is not lost to rounding in ``to_frac``.
_FRAC_SLACK = 1e-6

#: Any native region: a solid, or a composition of solids.
RegionLike = (
    Region
    | Cuboid
    | Sphere
    | HalfSpace
    | Parallelepiped
    | Cylinder
    | Ellipsoid
    | Polyhedron
    | SphereUnion
)


@dataclass(frozen=True)
class Site:
    """Lattice basis site in fractional coordinates.

    Attributes:
        label: Site identifier (e.g. ``"A"``, ``"B1"``).
        species: Chemical species or type name (e.g. ``"Ni"``, ``"Cl"``).
        fractional: Fractional coordinates ``(u, v, w)`` relative to the cell.
        charge: Partial charge (default ``0.0``).
        attrs: Optional auxiliary attributes.
    """

    label: str
    species: str
    fractional: tuple[float, float, float]
    charge: float = 0.0
    attrs: dict[str, Any] | None = None


class Lattice:
    """Bravais lattice = ``cell`` matrix + list of basis :class:`Site` objects.

    The cell matrix stores the three lattice vectors as rows::

        cell = [[a1x, a1y, a1z],
                [a2x, a2y, a2z],
                [a3x, a3y, a3z]]

    Construct directly with a matrix, or use :meth:`from_vectors` /
    :meth:`sc` / :meth:`bcc` / :meth:`fcc` / :meth:`rocksalt`.
    """

    def __init__(self, cell: ArrayLike, basis: list[Site] | None = None) -> None:
        cell_arr = np.asarray(cell, dtype=float)
        if cell_arr.shape != (3, 3):
            raise ValueError(f"cell must have shape (3, 3), got {cell_arr.shape}")
        self.cell = cell_arr
        self.basis: tuple[Site, ...] = tuple(basis or ())

    @property
    def a1(self) -> np.ndarray:
        return self.cell[0]

    @property
    def a2(self) -> np.ndarray:
        return self.cell[1]

    @property
    def a3(self) -> np.ndarray:
        return self.cell[2]

    @classmethod
    def from_vectors(
        cls,
        a1: ArrayLike,
        a2: ArrayLike,
        a3: ArrayLike,
        basis: list[Site] | None = None,
    ) -> Lattice:
        """Build a lattice from three lattice vectors."""
        return cls(np.stack([a1, a2, a3], axis=0), basis)

    def with_site(self, site: Site) -> Lattice:
        """Return a new lattice with ``site`` appended to the basis."""
        return Lattice(self.cell, [*self.basis, site])

    @property
    def box(self) -> Box:
        """The unit cell as a :class:`~molpy.Box` (lattice vectors as columns).

        Fractional ↔ Cartesian conversion is the box's:
        ``lattice.box.to_cart(frac)`` / ``lattice.box.to_frac(cart)``.
        """
        return Box(h=self.cell.T)

    def supercell(self, repeats: tuple[int, int, int]) -> Box:
        """The :class:`~molpy.Box` spanned by ``repeats`` unit cells.

        This is the simulation cell for a structure tiled with the same
        ``repeats``; ask the lattice for it rather than the built structure,
        which carries topology and chemistry but no cell of its own. The box
        matrix holds the lattice vectors as columns, so it is the transpose of
        the row-vector :attr:`cell`.
        """
        counts = _positive_repeats(repeats)
        return Box(h=(self.cell * counts.astype(float)[:, None]).T)

    @classmethod
    def sc(cls, a: float, species: str) -> Lattice:
        """Simple cubic lattice (1 atom / cell)."""
        return cls(a * np.eye(3), [Site("A", species, (0.0, 0.0, 0.0))])

    @classmethod
    def bcc(cls, a: float, species: str) -> Lattice:
        """Body-centered cubic lattice (2 atoms / cell)."""
        return cls(
            a * np.eye(3),
            [
                Site("A", species, (0.0, 0.0, 0.0)),
                Site("B", species, (0.5, 0.5, 0.5)),
            ],
        )

    @classmethod
    def fcc(cls, a: float, species: str) -> Lattice:
        """Face-centered cubic lattice (4 atoms / cell)."""
        return cls(
            a * np.eye(3),
            [
                Site("A", species, (0.0, 0.0, 0.0)),
                Site("B", species, (0.5, 0.5, 0.0)),
                Site("C", species, (0.5, 0.0, 0.5)),
                Site("D", species, (0.0, 0.5, 0.5)),
            ],
        )

    @classmethod
    def from_spacegroup(
        cls,
        cell: ArrayLike,
        sites: list[Site],
        spacegroup: SpaceGroup,
        *,
        symprec: float = 1e-5,
    ) -> Lattice:
        """Expand an asymmetric unit into a full-cell basis via a space group.

        A published crystal structure lists only the symmetry-inequivalent sites
        (the asymmetric unit); applying every operator of its space group fills
        the conventional cell. This is the native, zero-dependency path to a
        tiling-ready lattice from a CIF.

        Args:
            cell: ``(3, 3)`` cell matrix (lattice vectors as rows), e.g.
                ``a * np.eye(3)`` for a cubic cell of edge ``a``.
            sites: Asymmetric-unit basis sites (one :class:`Site` per
                inequivalent atom; ``fractional`` are its reduced coordinates).
            spacegroup: The :class:`~molpy.builder.SpaceGroup`, e.g.
                ``SpaceGroup.from_triplets(cif_symops)``.
            symprec: Fractional tolerance for collapsing images that land on a
                special position (passed to
                :meth:`~molpy.builder.SpaceGroup.equivalent_positions`).

        Returns:
            A :class:`Lattice` whose basis is every symmetry image of every input
            site; each image keeps its parent site's ``species``, ``charge``, and
            ``attrs``, and is labelled ``"<label>_<k>"``.
        """
        basis: list[Site] = []
        for site in sites:
            images = spacegroup.equivalent_positions(site.fractional, symprec=symprec)
            for k, frac in enumerate(images):
                basis.append(
                    Site(
                        label=f"{site.label}_{k}",
                        species=site.species,
                        fractional=(float(frac[0]), float(frac[1]), float(frac[2])),
                        charge=site.charge,
                        attrs=site.attrs,
                    )
                )
        return cls(cell, basis)

    @classmethod
    def rocksalt(cls, a: float, species_a: str, species_b: str) -> Lattice:
        """Rocksalt (NaCl) structure — two interpenetrating FCC sublattices."""
        basis = [
            Site("A1", species_a, (0.0, 0.0, 0.0)),
            Site("A2", species_a, (0.5, 0.5, 0.0)),
            Site("A3", species_a, (0.5, 0.0, 0.5)),
            Site("A4", species_a, (0.0, 0.5, 0.5)),
            Site("B1", species_b, (0.5, 0.0, 0.0)),
            Site("B2", species_b, (0.0, 0.5, 0.0)),
            Site("B3", species_b, (0.0, 0.0, 0.5)),
            Site("B4", species_b, (0.5, 0.5, 0.5)),
        ]
        return cls(a * np.eye(3), basis)

    def build(
        self,
        region: RegionLike | None = None,
        *,
        repeats: tuple[int, int, int] | None = None,
    ) -> Atomistic:
        """Tile this lattice and (optionally) clip to a Cartesian ``region``.

        Args:
            region: Geometric region in Cartesian space (e.g.
                ``mp.core.Cuboid``, ``mp.core.Sphere``, or any combination via
                ``& | ~``). Atoms outside the region are discarded.
            repeats: Number of unit cells along each lattice vector,
                ``(nx, ny, nz)``, tiled from the origin cell. If omitted, the
                tile is the range of cells, negative indices included, whose
                atoms can fall inside ``region.bounds()``. At least one of
                ``region`` or ``repeats`` must be provided.

        Returns:
            :class:`Atomistic` containing the kept atoms. The tiled super-cell is
            :meth:`Lattice.supercell`, which the caller sets on ``frame.box`` when
            the structure becomes a simulation.

        Raises:
            ValueError: If neither ``region`` nor ``repeats`` is given, if a
                repeat is not positive, or if ``repeats`` is omitted and
                ``region`` is unbounded.
        """
        if repeats is None and region is None:
            raise ValueError("Provide `region`, `repeats`, or both.")
        counts = None if repeats is None else _positive_repeats(repeats)

        out = Atomistic()

        if not self.basis:
            return out

        basis_fracs = np.array([s.fractional for s in self.basis], dtype=float)
        if counts is not None:
            lo, hi = np.zeros(3, dtype=int), counts
        else:
            lo, hi = _covering_cells(self.box, basis_fracs, region)
        cells = _cell_grid(lo, hi)  # (Nc, 3)
        fracs = (cells[:, None, :] + basis_fracs[None, :, :]).reshape(-1, 3)
        carts = self.box.to_cart(fracs)

        site_tiled = np.tile(np.array(self.basis, dtype=object), cells.shape[0])

        if region is not None:
            mask = region.contains(carts)
            carts = carts[mask]
            site_tiled = site_tiled[mask]

        if len(carts) == 0:
            return out

        # One Frame, one molrs call — not one def_atom per lattice point.
        # Canonical lattice fields win over optional site annotations, and every
        # site must declare the same annotation keys: a Frame has no "absent"
        # cell, and a filled-in default would be a guess.
        keys: set[str] = set()
        for site in self.basis:
            keys.update(site.attrs or {})
        for site in self.basis:
            missing = keys - set(site.attrs or {})
            if missing:
                raise ValueError(
                    f"site {site.label!r} lacks annotation(s) {sorted(missing)} "
                    "that other sites declare"
                )
        columns: dict[str, np.ndarray] = {
            key: np.array([site.attrs[key] for site in site_tiled])
            for key in sorted(keys)
        }
        columns.update(
            x=np.ascontiguousarray(carts[:, 0]),
            y=np.ascontiguousarray(carts[:, 1]),
            z=np.ascontiguousarray(carts[:, 2]),
            element=np.array([site.species for site in site_tiled]),
            charge=np.array([site.charge for site in site_tiled], dtype=float),
            label=np.array([site.label for site in site_tiled]),
        )
        frame = Frame()
        frame["atoms"] = columns
        return Atomistic.from_frame(frame)


def _positive_repeats(repeats: tuple[int, int, int]) -> np.ndarray:
    """``repeats`` as a ``(3,)`` int array, every count positive."""
    counts = np.array([int(r) for r in repeats], dtype=int)
    if counts.shape != (3,) or np.any(counts <= 0):
        raise ValueError(f"repeats must be three positive integers, got {repeats}")
    return counts


def _cell_grid(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Integer cell indices ``lo <= (i, j, k) < hi``, shape ``(Nc, 3)``."""
    axes = [np.arange(a, b) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _covering_cells(
    box: Box, basis_fracs: np.ndarray, region: RegionLike
) -> tuple[np.ndarray, np.ndarray]:
    """Cell range ``[lo, hi)`` whose atoms cover the region's bounding box.

    An atom of cell ``n`` sits at fractional ``n + b`` for a basis offset
    ``b``, so the cells that can reach the box's fractional extent
    ``[f_min, f_max]`` are ``floor(f_min - max b) <= n <= floor(f_max - min b)``,
    on every side of the origin.
    """
    bounds = np.asarray(region.bounds(), dtype=float)  # (3, 2): per-axis lo, hi
    if not np.all(np.isfinite(bounds)):
        raise ValueError(
            f"region is unbounded (bounds {bounds.tolist()}); "
            "pass `repeats` to choose the tile"
        )
    corners = np.array(list(itertools.product(*bounds)), dtype=float)  # (8, 3)
    frac = box.to_frac(corners)
    lo = np.floor(frac.min(axis=0) - basis_fracs.max(axis=0) - _FRAC_SLACK)
    hi = np.floor(frac.max(axis=0) - basis_fracs.min(axis=0) + _FRAC_SLACK) + 1
    return lo.astype(int), hi.astype(int)
