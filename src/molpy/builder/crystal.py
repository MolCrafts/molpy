"""
Crystal lattice builder module - LAMMPS-style crystal structure generator.

This module provides tools for creating crystal structures:
- Define Bravais lattices with basis sites
- Predefined common lattice types (SC, BCC, FCC, rocksalt)
- Define regions in lattice or Cartesian coordinates
- Efficient vectorized unit cell tiling and atom generation

Example:
    >>> lat = Lattice.cubic_fcc(a=3.52, species="Ni")
    >>> region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="lattice")
    >>> builder = CrystalBuilder(lat)
    >>> structure = builder.build_block(region)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from molpy import Atomistic, Box

# Coordinate system type
CoordSystem = Literal["lattice", "cartesian"]


@dataclass
class Site:
    """
    Lattice basis site in fractional coordinates.

    Attributes
    ----------
    label : str
        Site identifier or name (e.g., "A", "B1")
    species : str
        Chemical species or type name (e.g., "Ni", "Na", "Cl")
    frac : tuple[float, float, float]
        Fractional coordinates (u, v, w) relative to the Bravais cell, typically in [0, 1)
    charge : float, optional
        Charge, default is 0.0
    meta : dict[str, Any] | None, optional
        Optional metadata dictionary

    Examples
    --------
    >>> site = Site(label="A", species="Cu", frac=(0.0, 0.0, 0.0))
    >>> site_charged = Site(label="Na", species="Na", frac=(0.0, 0.0, 0.0), charge=1.0)
    """

    label: str
    species: str
    frac: tuple[float, float, float]
    charge: float = 0.0
    meta: dict[str, Any] | None = None


class Lattice:
    """
    Bravais lattice with basis sites.

    This class defines a crystal lattice structure, including lattice vectors
    and basis sites. Lattice vectors define the shape and size of the unit cell,
    while basis sites define the positions of atoms within the cell (in fractional
    coordinates).

    Parameters
    ----------
    a1, a2, a3 : np.ndarray
        Lattice vectors, each is a NumPy array of shape (3,)
    basis : list[Site]
        List of basis sites in fractional coordinates

    Attributes
    ----------
    a1, a2, a3 : np.ndarray
        Lattice vectors
    basis : list[Site]
        List of basis sites

    Examples
    --------
    >>> # Create simple cubic lattice
    >>> lat = Lattice.cubic_sc(a=2.0, species="Cu")
    >>>
    >>> # Create face-centered cubic lattice
    >>> lat = Lattice.cubic_fcc(a=3.52, species="Ni")
    >>>
    >>> # Create rocksalt structure
    >>> lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
    """

    def __init__(
        self, a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, basis: list[Site]
    ) -> None:
        """
        Initialize lattice

        Parameters
        ----------
        a1, a2, a3 : np.ndarray
            Lattice vectors of shape (3,)
        basis : list[Site]
            List of basis sites
        """
        self.a1 = np.asarray(a1, dtype=float)
        self.a2 = np.asarray(a2, dtype=float)
        self.a3 = np.asarray(a3, dtype=float)
        self.basis = list(basis)

        # Validate shape
        assert self.a1.shape == (3,), "a1 must be a 1D array of length 3"
        assert self.a2.shape == (3,), "a2 must be a 1D array of length 3"
        assert self.a3.shape == (3,), "a3 must be a 1D array of length 3"

    @property
    def cell(self) -> np.ndarray:
        """
        Return 3×3 cell matrix with lattice vectors as rows

        Returns
        -------
        np.ndarray
            Matrix of shape (3, 3), each row is a lattice vector [a1; a2; a3]
        """
        return np.stack([self.a1, self.a2, self.a3], axis=0)

    def add_site(self, site: Site) -> None:
        """
        Add a basis site

        Parameters
        ----------
        site : Site
            Basis site to add
        """
        self.basis.append(site)

    def frac_to_cart(self, frac: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates to Cartesian coordinates

        Fractional coordinates (u, v, w) represent position relative to lattice vectors:
        cart = u*a1 + v*a2 + w*a3 = frac @ cell

        Parameters
        ----------
        frac : np.ndarray
            Fractional coordinates of shape (N, 3) or (3,), containing (u, v, w) values

        Returns
        -------
        np.ndarray
            Cartesian coordinates, same shape as input

        Examples
        --------
        >>> lat = Lattice.cubic_sc(a=2.0, species="Cu")
        >>> frac = np.array([0.5, 0.5, 0.5])
        >>> cart = lat.frac_to_cart(frac)
        >>> print(cart)  # [1.0, 1.0, 1.0]
        """
        frac = np.asarray(frac, dtype=float)
        # Use single matrix multiplication: cart = frac @ cell
        return frac @ self.cell

    @classmethod
    def cubic_sc(cls, a: float, species: str) -> Lattice:
        """
        Create simple cubic (Simple Cubic, SC) lattice

        Simple cubic lattice is the simplest lattice type with one atom per unit cell.

        Parameters
        ----------
        a : float
            Lattice constant (in Å)
        species : str
            Atomic species (e.g., "Cu", "Fe")

        Returns
        -------
        Lattice
            Simple cubic lattice

        Examples
        --------
        >>> lat = Lattice.cubic_sc(a=2.0, species="Cu")
        >>> print(len(lat.basis))  # 1
        """
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
        a3 = np.array([0.0, 0.0, a])

        basis = [Site(label="A", species=species, frac=(0.0, 0.0, 0.0))]

        return cls(a1=a1, a2=a2, a3=a3, basis=basis)

    @classmethod
    def cubic_bcc(cls, a: float, species: str) -> Lattice:
        """
        Create body-centered cubic (Body-Centered Cubic, BCC) lattice

        Body-centered cubic lattice has two atoms per unit cell: one at corner and one at body center.

        Parameters
        ----------
        a : float
            Lattice constant (in Å)
        species : str
            Atomic species (e.g., "Fe", "W")

        Returns
        -------
        Lattice
            Body-centered cubic lattice

        Examples
        --------
        >>> lat = Lattice.cubic_bcc(a=3.0, species="Fe")
        >>> print(len(lat.basis))  # 2
        """
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
        a3 = np.array([0.0, 0.0, a])

        basis = [
            Site(label="A", species=species, frac=(0.0, 0.0, 0.0)),
            Site(label="B", species=species, frac=(0.5, 0.5, 0.5)),
        ]

        return cls(a1=a1, a2=a2, a3=a3, basis=basis)

    @classmethod
    def cubic_fcc(cls, a: float, species: str) -> Lattice:
        """
        Create face-centered cubic (Face-Centered Cubic, FCC) lattice

        Face-centered cubic lattice has four atoms per unit cell: one at corner and one at each face center.

        Parameters
        ----------
        a : float
            Lattice constant (in Å)
        species : str
            Atomic species (e.g., "Ni", "Cu", "Al")

        Returns
        -------
        Lattice
            Face-centered cubic lattice

        Examples
        --------
        >>> lat = Lattice.cubic_fcc(a=3.52, species="Ni")
        >>> print(len(lat.basis))  # 4
        """
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
        a3 = np.array([0.0, 0.0, a])

        basis = [
            Site(label="A", species=species, frac=(0.0, 0.0, 0.0)),
            Site(label="B", species=species, frac=(0.5, 0.5, 0.0)),
            Site(label="C", species=species, frac=(0.5, 0.0, 0.5)),
            Site(label="D", species=species, frac=(0.0, 0.5, 0.5)),
        ]

        return cls(a1=a1, a2=a2, a3=a3, basis=basis)

    @classmethod
    def rocksalt(cls, a: float, species_a: str, species_b: str) -> Lattice:
        """
        Create rocksalt (NaCl) structure

        Rocksalt structure consists of two interpenetrating FCC sublattices.
        Each unit cell contains 4 A atoms and 4 B atoms.

        Parameters
        ----------
        a : float
            Lattice constant (in Å)
        species_a : str
            First atomic species (e.g., "Na")
        species_b : str
            Second atomic species (e.g., "Cl")

        Returns
        -------
        Lattice
            Rocksalt structure lattice

        Examples
        --------
        >>> lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
        >>> print(len(lat.basis))  # 8
        """
        a1 = np.array([a, 0.0, 0.0])
        a2 = np.array([0.0, a, 0.0])
        a3 = np.array([0.0, 0.0, a])

        # FCC sites for species_a
        basis_a = [
            Site(label="A1", species=species_a, frac=(0.0, 0.0, 0.0)),
            Site(label="A2", species=species_a, frac=(0.5, 0.5, 0.0)),
            Site(label="A3", species=species_a, frac=(0.5, 0.0, 0.5)),
            Site(label="A4", species=species_a, frac=(0.0, 0.5, 0.5)),
        ]

        # Offset FCC sites for species_b (shifted by 0.5 along x direction)
        basis_b = [
            Site(label="B1", species=species_b, frac=(0.5, 0.0, 0.0)),
            Site(label="B2", species=species_b, frac=(0.0, 0.5, 0.0)),
            Site(label="B3", species=species_b, frac=(0.0, 0.0, 0.5)),
            Site(label="B4", species=species_b, frac=(0.5, 0.5, 0.5)),
        ]

        basis = basis_a + basis_b

        return cls(a1=a1, a2=a2, a3=a3, basis=basis)


class Region(ABC):
    """
    Abstract geometric region class

    Define a spatial region that can be represented in lattice or Cartesian coordinates.

    Parameters
    ----------
    coord_system : CoordSystem
        Coordinate system, "lattice" or "cartesian"
        - "lattice": point coordinates in lattice units
        - "cartesian": point coordinates in Cartesian coordinates (Å)

    Notes
    -----
    Subclasses must implement `contains_mask` method using NumPy
    vectorized operations to efficiently check if multiple points are in the region.
    """

    def __init__(self, coord_system: CoordSystem = "lattice") -> None:
        """
        Initialize region

        Parameters
        ----------
        coord_system : CoordSystem, optional
            Coordinate system, default is "lattice"
        """
        self.coord_system = coord_system

    @abstractmethod
    def contains_mask(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are in the region (vectorized)

        Parameters
        ----------
        points : np.ndarray
            Point coordinates array of shape (N, 3)

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,), True indicates point is in region

        Notes
        -----
        - If coord_system == "lattice": points are in lattice units
        - If coord_system == "cartesian": points are in Cartesian coordinates
        - Must use vectorized operations, no Python loops
        """
        pass


class BlockRegion(Region):
    """
    Axis-aligned box region

    Define a box region specified by x, y, z ranges.

    Parameters
    ----------
    xmin, xmax : float
        x-direction range [xmin, xmax]
    ymin, ymax : float
        y-direction range [ymin, ymax]
    zmin, zmax : float
        z-direction range [zmin, zmax]
    coord_system : CoordSystem, optional
        Coordinate system, default is "lattice"

    Examples
    --------
    >>> # Region in lattice coordinates
    >>> region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="lattice")
    >>>
    >>> # Region in Cartesian coordinates
    >>> region = BlockRegion(0, 30, 0, 30, 0, 30, coord_system="cartesian")
    """

    def __init__(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        zmin: float,
        zmax: float,
        coord_system: CoordSystem = "lattice",
    ) -> None:
        """
        Initialize box region

        Parameters
        ----------
        xmin, xmax : float
            x-direction range
        ymin, ymax : float
            y-direction range
        zmin, zmax : float
            z-direction range
        coord_system : CoordSystem, optional
            Coordinate system
        """
        super().__init__(coord_system)
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.zmin = zmin
        self.zmax = zmax

    def contains_mask(self, points: np.ndarray) -> np.ndarray:
        """
        Check if points are in the box (vectorized)

        Parameters
        ----------
        points : np.ndarray
            Point coordinates array of shape (N, 3)

        Returns
        -------
        np.ndarray
            Boolean array of shape (N,)

        Examples
        --------
        >>> region = BlockRegion(0, 10, 0, 10, 0, 10)
        >>> points = np.array([[5, 5, 5], [15, 5, 5]])
        >>> mask = region.contains_mask(points)
        >>> print(mask)  # [True, False]
        """
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points.reshape(1, -1)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Vectorized boundary check
        mask = (
            (self.xmin <= x)
            & (x <= self.xmax)
            & (self.ymin <= y)
            & (y <= self.ymax)
            & (self.zmin <= z)
            & (z <= self.zmax)
        )

        return mask


class CrystalBuilder:
    """
    Crystal structure builder

    Efficiently generate crystal structures using NumPy vectorized operations.
    Supports tiling lattices and creating atoms in specified regions.

    Parameters
    ----------
    lattice : Lattice
        Lattice definition to use

    Examples
    --------
    >>> # Create a simple FCC structure
    >>> lat = Lattice.cubic_fcc(a=3.52, species="Ni")
    >>> region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="lattice")
    >>> builder = CrystalBuilder(lat)
    >>> structure = builder.build_block(region)
    >>> print(len(structure.atoms))
    """

    def __init__(self, lattice: Lattice) -> None:
        """
        Initialize crystal builder

        Parameters
        ----------
        lattice : Lattice
            Lattice definition
        """
        self.lattice = lattice

    def build_block(
        self,
        region: BlockRegion,
        *,
        i_range: range | None = None,
        j_range: range | None = None,
        k_range: range | None = None,
    ) -> Atomistic:
        """
        Build crystal structure within a box region

        This method efficiently generates crystal structures using vectorized operations:
        1. Determine cell index ranges to tile
        2. Use NumPy meshgrid and broadcasting to generate all atom positions
        3. Apply region filtering
        4. Create and return Atomistic structure

        Parameters
        ----------
        region : BlockRegion
            Box defining the region for atom generation
        i_range, j_range, k_range : range | None, optional
            Explicitly specify cell index ranges. If not provided:
            - For "lattice" coordinate system: inferred from region boundaries
            - For "cartesian" coordinate system: must be provided, otherwise raises error

        Returns
        -------
        Atomistic
            Generated crystal structure containing atoms and box information

        Raises
        ------
        ValueError
            If coord_system == "cartesian" and explicit ranges are not provided

        Examples
        --------
        >>> # Using lattice coordinates (auto-infer ranges)
        >>> lat = Lattice.cubic_sc(a=2.0, species="Cu")
        >>> region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="lattice")
        >>> builder = CrystalBuilder(lat)
        >>> structure = builder.build_block(region)
        >>>
        >>> # Using explicit ranges
        >>> structure = builder.build_block(
        ...     region,
        ...     i_range=range(0, 5),
        ...     j_range=range(0, 5),
        ...     k_range=range(0, 5)
        ... )
        >>>
        >>> # Cartesian coordinates (must provide ranges)
        >>> region_cart = BlockRegion(0, 20, 0, 20, 0, 20, coord_system="cartesian")
        >>> structure = builder.build_block(
        ...     region_cart,
        ...     i_range=range(0, 10),
        ...     j_range=range(0, 10),
        ...     k_range=range(0, 10)
        ... )

        Notes
        -----
        - This method uses no Python loops, fully based on NumPy vectorized operations
        - Generated structure contains:
          - Atom positions (Cartesian coordinates)
          - Atom species
          - Box information (lattice vectors)
        - For empty basis (no basis sites), returns empty Atomistic structure
        """
        basis_sites = self.lattice.basis

        # If no basis sites, return empty structure
        if len(basis_sites) == 0:
            structure = Atomistic()
            structure["box"] = Box(matrix=self.lattice.cell)
            return structure

        # Step 1: Determine cell index ranges
        if i_range is not None and j_range is not None and k_range is not None:
            # Use explicitly provided ranges
            i_arr = np.array(list(i_range))
            j_arr = np.array(list(j_range))
            k_arr = np.array(list(k_range))
        elif region.coord_system == "lattice":
            # Infer ranges from lattice coordinate region
            i_min = int(np.floor(region.xmin))
            i_max = int(np.ceil(region.xmax))
            j_min = int(np.floor(region.ymin))
            j_max = int(np.ceil(region.ymax))
            k_min = int(np.floor(region.zmin))
            k_max = int(np.ceil(region.zmax))

            i_arr = np.arange(i_min, i_max)
            j_arr = np.arange(j_min, j_max)
            k_arr = np.arange(k_min, k_max)
        else:
            # Cartesian coordinate system requires explicit ranges
            raise ValueError(
                "For cartesian coordinate system, you must provide explicit "
                "i_range, j_range, and k_range parameters."
            )

        # Step 2: Build cell index grid using NumPy
        I, J, K = np.meshgrid(i_arr, j_arr, k_arr, indexing="ij")
        cells = np.stack([I, J, K], axis=-1).reshape(-1, 3)  # (Nc, 3)

        # Step 3: Combine cells and basis via broadcasting (no Python loops)
        basis_fracs = np.array(
            [list(s.frac) for s in basis_sites], dtype=float
        )  # (Nb, 3)

        # Broadcasting: cells[:, None, :] is (Nc, 1, 3), basis_fracs[None, :, :] is (1, Nb, 3)
        # Result is (Nc, Nb, 3)
        frac_lattice = cells[:, None, :] + basis_fracs[None, :, :]  # (Nc, Nb, 3)
        frac_lattice = frac_lattice.reshape(-1, 3)  # (N, 3) where N = Nc * Nb

        # Step 4: Region filtering + coordinate system conversion
        if region.coord_system == "lattice":
            # Filter in lattice coordinates
            mask = region.contains_mask(frac_lattice)
            frac_selected = frac_lattice[mask]

            # Convert to Cartesian coordinates
            cart_selected = self.lattice.frac_to_cart(frac_selected)
        else:
            # coord_system == "cartesian"
            # First convert all coordinates to Cartesian
            cart_all = self.lattice.frac_to_cart(frac_lattice)
            mask = region.contains_mask(cart_all)
            cart_selected = cart_all[mask]
            frac_selected = frac_lattice[mask]

        # Step 5: Get species and site info via vectorization
        Nc = cells.shape[0]
        species_array = np.array(
            [s.species for s in basis_sites], dtype=object
        )  # (Nb,)
        site_array = np.array(basis_sites, dtype=object)  # (Nb,)

        # Tile to match each cell
        species_tiled = np.tile(species_array, Nc)  # (N,)
        site_tiled = np.tile(site_array, Nc)  # (N,)

        # Apply filtering
        species_selected = species_tiled[mask]  # (M,)
        site_selected = site_tiled[mask]  # (M,)

        # Step 6: Build project's Atomistic structure type
        structure = Atomistic()

        # Add atoms
        for i in range(len(cart_selected)):
            xyz = cart_selected[i]
            species = species_selected[i]
            site = site_selected[i]

            structure.def_atom(
                xyz=xyz.tolist(),
                symbol=species,
                charge=site.charge,
                label=site.label,
            )

        # Set box (using lattice matrix)
        structure["box"] = Box(matrix=self.lattice.cell)

        return structure
