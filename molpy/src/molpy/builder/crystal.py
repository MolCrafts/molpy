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
    >>> # or expand a CIF's asymmetric unit by its own operators:
    >>> salt = mp.builder.Lattice.from_cif(mp.io.read_cif("NaCl.cif"))  # doctest: +SKIP
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from molrs.core import (
    Atomistic,
    Block,
    Box,
    Cell,
    Element,
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
    keys,
)
from numpy.typing import ArrayLike

from molpy.builder.symmetry import SpaceGroup

__all__ = ["Lattice", "Site"]

#: Fractional slack on the inferred tile, so an atom on the region's
#: boundary is not lost to rounding in ``to_frac``.
_FRAC_SLACK = 1e-6
#: An occupancy this close below 1 still counts as a fully occupied site.
_OCC_TOL = 1e-6

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
    """Bravais lattice = a :class:`~molpy.Box` unit cell + basis :class:`Site` objects.

    The box is molrs's: its cell matrix ``box.matrix`` holds the lattice vectors as
    columns, and it owns fractional ↔ Cartesian conversion
    (``lattice.box.to_cart(frac)`` / ``lattice.box.to_frac(cart)``).

    Construct directly with a box, or use :meth:`from_vectors` /
    :meth:`from_cif` / :meth:`from_spacegroup` / :meth:`sc` / :meth:`bcc` /
    :meth:`fcc` / :meth:`rocksalt`.

    Raises:
        TypeError: If ``box`` is not a :class:`~molpy.Box`.
        ValueError: If ``box`` has no cell (``box.is_cell_defined`` is false).
    """

    def __init__(self, box: Box, basis: list[Site] | None = None) -> None:
        if not isinstance(box, Box):
            raise TypeError(
                f"Lattice takes a molpy.Box, got {type(box).__name__}; "
                "build one from vectors with Lattice.from_vectors(a1, a2, a3)"
            )
        if not box.is_cell_defined:
            raise ValueError("Lattice needs a box with a defined cell")
        self.box = box
        self.basis: tuple[Site, ...] = tuple(basis or ())

    @property
    def a1(self) -> np.ndarray:
        """First lattice vector (column 0 of ``box.matrix``)."""
        return self.box.matrix[:, 0]

    @property
    def a2(self) -> np.ndarray:
        """Second lattice vector (column 1 of ``box.matrix``)."""
        return self.box.matrix[:, 1]

    @property
    def a3(self) -> np.ndarray:
        """Third lattice vector (column 2 of ``box.matrix``)."""
        return self.box.matrix[:, 2]

    @classmethod
    def from_vectors(
        cls,
        a1: ArrayLike,
        a2: ArrayLike,
        a3: ArrayLike,
        basis: list[Site] | None = None,
    ) -> Lattice:
        """Build a lattice from three lattice vectors (the box's columns)."""
        h = np.column_stack([np.asarray(a, dtype=float) for a in (a1, a2, a3)])
        return cls(Box(Cell.matrix(h)), basis)

    def with_site(self, site: Site) -> Lattice:
        """Return a new lattice with ``site`` appended to the basis."""
        return Lattice(self.box, [*self.basis, site])

    def supercell(self, repeats: tuple[int, int, int]) -> Box:
        """The :class:`~molpy.Box` spanned by ``repeats`` unit cells.

        This is the simulation cell for a structure tiled with the same
        ``repeats``; ask the lattice for it rather than the built structure,
        which carries topology and chemistry but no cell of its own. Each
        lattice vector (column of ``box.matrix``) is scaled by its repeat; the
        origin and periodicity are the unit cell's.
        """
        counts = _positive_repeats(repeats)
        return self.box.transformed(np.diag(counts.astype(float)))

    @classmethod
    def sc(cls, a: float, species: str) -> Lattice:
        """Simple cubic lattice (1 atom / cell)."""
        return cls(Box(Cell.lengths([a] * 3)), [Site("A", species, (0.0, 0.0, 0.0))])

    @classmethod
    def bcc(cls, a: float, species: str) -> Lattice:
        """Body-centered cubic lattice (2 atoms / cell)."""
        return cls(
            Box(Cell.lengths([a] * 3)),
            [
                Site("A", species, (0.0, 0.0, 0.0)),
                Site("B", species, (0.5, 0.5, 0.5)),
            ],
        )

    @classmethod
    def fcc(cls, a: float, species: str) -> Lattice:
        """Face-centered cubic lattice (4 atoms / cell)."""
        return cls(
            Box(Cell.lengths([a] * 3)),
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
        box: Box,
        sites: list[Site],
        spacegroup: SpaceGroup,
        *,
        symprec: float = 1e-5,
    ) -> Lattice:
        """Expand an asymmetric unit into a full-cell basis via a space group.

        A published crystal structure lists only the symmetry-inequivalent sites
        (the asymmetric unit); applying every operator of its space group fills
        the conventional cell. For a CIF, :meth:`from_cif` does this from the
        frame :func:`molpy.io.read_cif` returns.

        Args:
            box: The unit cell, e.g. ``mp.Box.cube(a)`` or a CIF frame's
                ``frame.box``.
            sites: Asymmetric-unit basis sites (one :class:`Site` per
                inequivalent atom; ``fractional`` are its reduced coordinates).
            spacegroup: The :class:`~molpy.builder.SpaceGroup`, e.g.
                ``SpaceGroup.from_generators([...])``.
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
        return cls(box, basis)

    @classmethod
    def from_cif(cls, frame: Frame, *, symprec: float = 1e-5) -> Lattice:
        """The full-cell lattice of a CIF read by :func:`molpy.io.read_cif`.

        The cell is ``frame.box``; the asymmetric unit is the ``atoms`` block
        (label from ``name``, species from ``element``, fractional coordinates
        from ``frame.box.to_frac``); the space group is the file's operator
        loop, which ``read_cif`` keeps verbatim in ``frame.meta["symops"]``.
        Every atom is expanded by :meth:`from_spacegroup`.

        Args:
            frame: A frame from :func:`molpy.io.read_cif`.
            symprec: Fractional tolerance for collapsing images on a special
                position.

        Returns:
            A :class:`Lattice` on ``frame.box`` with the whole conventional cell
            as its basis; each image is labelled ``"<name>_<k>"``.

        Raises:
            ValueError: If the frame has no box or no ``symops`` meta (a CIF
                with no operator loop is not taken to be P1: state ``x,y,z``
                in the file), if the atoms block lacks ``name``, ``element`` or
                coordinates, or if any site is partially occupied. The lattice
                places one atom per site, so a disordered structure would be
                built fully occupied; resolve the disorder before building.
        """
        box = frame.box
        if box is None or not box.is_cell_defined:
            raise ValueError(
                "frame has no box (the CIF states no _cell_length_* / "
                "_cell_angle_* cell); Lattice.from_cif needs the unit cell"
            )
        symops_key = str(keys.SYMOPS)
        if symops_key not in frame.meta:
            raise ValueError(
                f"frame.meta has no {symops_key!r} (the CIF has no "
                "_space_group_symop_operation_xyz or _symmetry_equiv_pos_as_xyz "
                "loop); refusing to assume P1. For a P1 structure, list 'x,y,z' "
                "in the file or use Lattice.from_spacegroup"
            )
        atoms = frame["atoms"]
        required = [str(k) for k in (keys.NAME, keys.ELEMENT, *keys.COORDS)]
        missing = [k for k in required if k not in atoms]
        if missing:
            raise ValueError(f"CIF atoms block lacks column(s) {missing}")
        names = [str(n) for n in atoms[str(keys.NAME)]]
        _reject_partial_occupancy(atoms, names)
        xyz = np.column_stack([atoms[str(k)] for k in keys.COORDS])
        fractional = box.to_frac(np.asarray(xyz, dtype=float))
        sites = [
            Site(label, str(species), (float(u), float(v), float(w)))
            for label, species, (u, v, w) in zip(
                names, atoms[str(keys.ELEMENT)], fractional, strict=True
            )
        ]
        group = SpaceGroup.from_triplets(frame.meta[symops_key])
        return cls.from_spacegroup(box, sites, group, symprec=symprec)

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
        return cls(Box(Cell.lengths([a] * 3)), basis)

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
        Element.fill_atomic_numbers(frame)
        return Atomistic.from_frame(frame)


def _reject_partial_occupancy(atoms: Block, names: list[str]) -> None:
    """Raise if any site of a CIF atoms block is less than fully occupied."""
    column = str(keys.OCCUPANCY)
    if column not in atoms:
        return
    occupancy = np.asarray(atoms[column], dtype=float)
    partial = [
        n for n, occ in zip(names, occupancy, strict=True) if occ < 1.0 - _OCC_TOL
    ]
    if partial:
        raise ValueError(
            f"site(s) {partial} have partial occupancy (< 1); Lattice places one "
            "atom per site and cannot represent disorder. Resolve the disorder "
            "in the CIF (keep one component at occupancy 1) before building"
        )


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
