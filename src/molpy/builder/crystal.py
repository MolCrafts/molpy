"""Crystal lattice builder.

Tile a Bravais lattice over a range of unit cells and (optionally) clip the
result to a geometric :class:`molpy.core.region.Region`.

Example:
    >>> from molpy.builder import Lattice, build_crystal
    >>> from molpy.core.region import BoxRegion
    >>> lat = Lattice.fcc(a=3.52, species="Ni")
    >>> structure = build_crystal(lat, repeats=(4, 4, 4))
    >>> # or clip a 30 Å cube out of a larger tile:
    >>> structure = build_crystal(lat, BoxRegion(lengths=[30, 30, 30]))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from molpy.core.atomistic import Atomistic
from molpy.core.box import Box
from molpy.core.region import Region
from molpy.builder.symmetry import SpaceGroup

__all__ = ["Lattice", "Site", "SpaceGroup", "build_crystal"]


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

    def frac_to_cart(self, frac: ArrayLike) -> np.ndarray:
        """Fractional → Cartesian: ``frac @ cell``."""
        return np.asarray(frac, dtype=float) @ self.cell

    def cart_to_frac(self, cart: ArrayLike) -> np.ndarray:
        """Cartesian → fractional: ``cart @ cell⁻¹``."""
        return np.asarray(cart, dtype=float) @ np.linalg.inv(self.cell)

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
            spacegroup: The :class:`~molpy.builder.symmetry.SpaceGroup`, e.g.
                ``SpaceGroup.from_triplets(cif_symops)``.
            symprec: Fractional tolerance for collapsing images that land on a
                special position (passed to
                :meth:`~molpy.builder.symmetry.SpaceGroup.equivalent_positions`).

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


def build_crystal(
    lattice: Lattice,
    region: Region | None = None,
    *,
    repeats: tuple[int, int, int] | None = None,
) -> Atomistic:
    """Tile ``lattice`` and (optionally) clip to a Cartesian ``region``.

    Args:
        lattice: Bravais lattice with basis sites.
        region: Geometric region in Cartesian space (e.g.
            :class:`molpy.core.region.BoxRegion`, ``SphereRegion``, or any
            ``Region`` combination via ``& | ~``). Atoms outside the region
            are discarded.
        repeats: Number of unit cells along each lattice vector,
            ``(nx, ny, nz)``. If omitted, the tile range is inferred from
            ``region.bounds``. At least one of ``region`` or ``repeats``
            must be provided.

    Returns:
        :class:`Atomistic` containing the kept atoms and a ``box`` set to the
        full tiled super-cell (``cell`` scaled row-wise by ``repeats``).
    """
    if region is None and repeats is None:
        raise ValueError("Provide `region`, `repeats`, or both.")

    if repeats is None:
        repeats = _infer_repeats(lattice, region.bounds)

    nx, ny, nz = (int(r) for r in repeats)
    if nx <= 0 or ny <= 0 or nz <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    super_cell = lattice.cell * np.array([nx, ny, nz], dtype=float)[:, None]

    out = Atomistic()
    out["box"] = Box(matrix=super_cell)

    if not lattice.basis:
        return out

    cells = _cell_grid(nx, ny, nz)  # (Nc, 3)
    basis_fracs = np.array([s.fractional for s in lattice.basis], dtype=float)
    fracs = (cells[:, None, :] + basis_fracs[None, :, :]).reshape(-1, 3)
    carts = lattice.frac_to_cart(fracs)

    site_tiled = np.tile(np.array(lattice.basis, dtype=object), cells.shape[0])

    if region is not None:
        mask = region.isin(carts)
        carts = carts[mask]
        site_tiled = site_tiled[mask]

    for xyz, site in zip(carts, site_tiled):
        out.def_atom(
            xyz=xyz.tolist(),
            element=site.species,
            charge=site.charge,
            label=site.label,
        )
    return out


def _cell_grid(nx: int, ny: int, nz: int) -> np.ndarray:
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    return np.stack([i, j, k], axis=-1).reshape(-1, 3)


def _infer_repeats(lattice: Lattice, bounds: np.ndarray) -> tuple[int, int, int]:
    """Smallest ``(nx, ny, nz)`` (starting at origin) covering ``bounds`` AABB."""
    lo, hi = bounds[0], bounds[1]
    corners = (
        np.array(
            np.meshgrid([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]], indexing="ij")
        )
        .reshape(3, -1)
        .T
    )
    frac_extent = lattice.cart_to_frac(corners).max(axis=0)
    repeats = np.ceil(np.maximum(frac_extent, 0)).astype(int)
    repeats = np.maximum(repeats, 1)
    return int(repeats[0]), int(repeats[1]), int(repeats[2])
