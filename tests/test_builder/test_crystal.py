from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molpy import Atomistic, Box
from molpy.builder import BoxRegion, Lattice, Site, SphereRegion, build_crystal


class TestSite:
    def test_site_creation(self):
        site = Site(label="A", species="Ni", fractional=(0.0, 0.0, 0.0))
        assert site.label == "A"
        assert site.species == "Ni"
        assert site.fractional == (0.0, 0.0, 0.0)
        assert site.charge == 0.0
        assert site.attrs is None

    def test_site_with_charge(self):
        site = Site(label="Na", species="Na", fractional=(0.0, 0.0, 0.0), charge=1.0)
        assert site.charge == 1.0

    def test_site_with_attrs(self):
        attrs = {"tag": "corner"}
        site = Site(label="A", species="C", fractional=(0.0, 0.0, 0.0), attrs=attrs)
        assert site.attrs == attrs

    def test_site_is_frozen(self):
        site = Site(label="A", species="C", fractional=(0.0, 0.0, 0.0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            site.label = "B"  # type: ignore[misc]


class TestLattice:
    def test_lattice_creation_from_matrix(self):
        cell = np.eye(3)
        lattice = Lattice(cell=cell, basis=[])

        assert np.allclose(lattice.cell, cell)
        assert lattice.basis == ()

    def test_lattice_rejects_bad_shape(self):
        with pytest.raises(ValueError, match="shape"):
            Lattice(cell=np.eye(2))

    def test_lattice_vector_accessors(self):
        cell = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        lattice = Lattice(cell=cell)

        assert np.allclose(lattice.a1, [3.0, 0.0, 0.0])
        assert np.allclose(lattice.a2, [0.0, 4.0, 0.0])
        assert np.allclose(lattice.a3, [0.0, 0.0, 5.0])

    def test_from_vectors(self):
        a1 = np.array([3.0, 0.0, 0.0])
        a2 = np.array([0.0, 3.0, 0.0])
        a3 = np.array([0.0, 0.0, 3.0])
        lattice = Lattice.from_vectors(a1, a2, a3, basis=[])

        assert lattice.cell.shape == (3, 3)
        assert np.allclose(lattice.cell[0], a1)
        assert np.allclose(lattice.cell[1], a2)
        assert np.allclose(lattice.cell[2], a3)

    def test_with_site_is_immutable(self):
        base = Lattice(cell=np.eye(3))
        site = Site(label="A", species="C", fractional=(0.0, 0.0, 0.0))
        extended = base.with_site(site)

        assert base.basis == ()
        assert extended.basis == (site,)
        assert extended is not base

    def test_frac_to_cart_single(self):
        cell = np.diag([3.0, 4.0, 5.0])
        lattice = Lattice(cell=cell)

        cart = lattice.frac_to_cart(np.array([0.5, 0.5, 0.5]))
        assert np.allclose(cart, [1.5, 2.0, 2.5])

    def test_frac_to_cart_multiple(self):
        lattice = Lattice(cell=2.0 * np.eye(3))

        frac = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]]
        )
        cart = lattice.frac_to_cart(frac)
        expected = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
        )
        assert np.allclose(cart, expected)

    def test_cart_to_frac_inverts_frac_to_cart(self):
        cell = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        lattice = Lattice(cell=cell)

        frac = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
        cart = lattice.frac_to_cart(frac)
        recovered = lattice.cart_to_frac(cart)
        assert np.allclose(recovered, frac)

    def test_sc(self):
        lat = Lattice.sc(a=2.0, species="Cu")

        assert np.allclose(lat.cell, 2.0 * np.eye(3))
        assert len(lat.basis) == 1
        assert lat.basis[0].species == "Cu"
        assert lat.basis[0].fractional == (0.0, 0.0, 0.0)

    def test_bcc(self):
        lat = Lattice.bcc(a=3.0, species="Fe")

        assert np.allclose(lat.cell, 3.0 * np.eye(3))
        assert len(lat.basis) == 2

        fracs = [site.fractional for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.5) in fracs
        assert all(site.species == "Fe" for site in lat.basis)

    def test_fcc(self):
        lat = Lattice.fcc(a=3.52, species="Ni")

        assert len(lat.basis) == 4
        fracs = [site.fractional for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.0) in fracs
        assert (0.5, 0.0, 0.5) in fracs
        assert (0.0, 0.5, 0.5) in fracs
        assert all(site.species == "Ni" for site in lat.basis)

    def test_rocksalt(self):
        lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")

        assert len(lat.basis) == 8
        na_count = sum(1 for s in lat.basis if s.species == "Na")
        cl_count = sum(1 for s in lat.basis if s.species == "Cl")
        assert na_count == 4
        assert cl_count == 4

        na_fracs = [s.fractional for s in lat.basis if s.species == "Na"]
        assert (0.0, 0.0, 0.0) in na_fracs
        assert (0.5, 0.5, 0.0) in na_fracs

        cl_fracs = [s.fractional for s in lat.basis if s.species == "Cl"]
        assert (0.5, 0.0, 0.0) in cl_fracs
        assert (0.0, 0.5, 0.0) in cl_fracs


class TestBuildCrystalRepeats:
    def test_sc_repeats(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        assert isinstance(structure, Atomistic)
        assert len(list(structure.atoms)) == 8
        assert all(s == "Cu" for s in structure.symbols)

    def test_bcc_repeats(self):
        lat = Lattice.bcc(a=2.0, species="Fe")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 16
        assert all(s == "Fe" for s in structure.symbols)

    def test_fcc_repeats(self):
        lat = Lattice.fcc(a=3.52, species="Ni")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 32
        assert all(s == "Ni" for s in structure.symbols)

    def test_rocksalt_repeats(self):
        lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 64
        na = sum(1 for s in structure.symbols if s == "Na")
        cl = sum(1 for s in structure.symbols if s == "Cl")
        assert na == 32
        assert cl == 32

    def test_empty_basis(self):
        lat = Lattice(cell=np.eye(3))
        structure = build_crystal(lat, repeats=(2, 2, 2))

        assert isinstance(structure, Atomistic)
        assert len(list(structure.atoms)) == 0

    def test_super_cell_box(self):
        lat = Lattice.sc(a=3.0, species="Cu")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        box = structure["box"]
        assert isinstance(box, Box)
        assert np.allclose(box.matrix, 6.0 * np.eye(3))

    def test_positions(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        structure = build_crystal(lat, repeats=(2, 2, 2))

        positions = structure.xyz
        expected = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0, 0, 2],
                [2, 2, 0],
                [2, 0, 2],
                [0, 2, 2],
                [2, 2, 2],
            ],
            dtype=float,
        )
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        exp_order = np.lexsort((expected[:, 2], expected[:, 1], expected[:, 0]))
        assert np.allclose(positions[order], expected[exp_order])


class TestBuildCrystalRegion:
    def test_box_region_infers_repeats(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        structure = build_crystal(lat, BoxRegion(lengths=[3.0, 3.0, 3.0]))

        # cells inferred = ceil(3/2)=2 along each axis → 8 atoms generated,
        # all inside the [0,3]³ region (positions ∈ {0, 2}).
        assert len(list(structure.atoms)) == 8

    def test_box_region_clips_extra_atoms(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        # Force a 3-cell tile but clip to a 3 Å box: corner atoms at x=4 etc.
        # are filtered out.
        structure = build_crystal(
            lat, BoxRegion(lengths=[3.0, 3.0, 3.0]), repeats=(3, 3, 3)
        )

        assert len(list(structure.atoms)) == 8

    def test_sphere_region(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        structure = build_crystal(
            lat,
            SphereRegion(radius=1.5, center=[1.5, 1.5, 1.5]),
            repeats=(4, 4, 4),
        )

        # Every atom must lie within the sphere.
        positions = structure.xyz
        center = np.array([1.5, 1.5, 1.5])
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.all(distances <= 1.5 + 1e-9)
        assert len(positions) > 0

    def test_combined_regions(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        cube = BoxRegion(lengths=[3.0, 3.0, 3.0])
        sphere = SphereRegion(radius=1.5, center=[1.5, 1.5, 1.5])
        # Intersection: atoms in both.
        structure = build_crystal(lat, cube & sphere, repeats=(4, 4, 4))

        positions = structure.xyz
        center = np.array([1.5, 1.5, 1.5])
        in_sphere = np.linalg.norm(positions - center, axis=1) <= 1.5 + 1e-9
        in_box = np.all((positions >= 0) & (positions <= 3.0), axis=1)
        assert np.all(in_sphere & in_box)

    def test_requires_region_or_repeats(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="region.*repeats"):
            build_crystal(lat)

    def test_rejects_non_positive_repeats(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="repeats must be positive"):
            build_crystal(lat, repeats=(0, 1, 1))
