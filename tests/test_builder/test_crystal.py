from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from molpy.core import Atomistic, Box, Cuboid, Sphere
from molpy.builder import Lattice, Site


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

    def test_site_attrs_are_copied_to_built_atoms(self):
        site = Site(
            label="A",
            species="C",
            fractional=(0.0, 0.0, 0.0),
            attrs={"domain": "wall"},
        )
        structure = Lattice(np.eye(3), [site]).build(repeats=(1, 1, 1))
        assert structure.atoms[0].get("domain") == "wall"

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

    def test_box_to_cart_single(self):
        cell = np.diag([3.0, 4.0, 5.0])
        lattice = Lattice(cell=cell)

        cart = lattice.box.to_cart(np.array([[0.5, 0.5, 0.5]]))[0]
        assert np.allclose(cart, [1.5, 2.0, 2.5])

    def test_box_to_cart_multiple(self):
        lattice = Lattice(cell=2.0 * np.eye(3))

        frac = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]]
        )
        cart = lattice.box.to_cart(frac)
        expected = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
        )
        assert np.allclose(cart, expected)

    def test_box_to_frac_inverts_to_cart_and_rows_are_lattice_vectors(self):
        cell = np.array([[2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        lattice = Lattice(cell=cell)

        frac = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
        cart = lattice.box.to_cart(frac)
        recovered = lattice.box.to_frac(cart)
        assert np.allclose(recovered, frac)
        # The cell's rows are the lattice vectors: cart = frac @ cell.
        assert np.allclose(cart, frac @ cell)

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
        structure = lat.build(repeats=(2, 2, 2))

        assert isinstance(structure, Atomistic)
        assert len(list(structure.atoms)) == 8
        assert all(s == "Cu" for s in structure.atoms["element"])

    def test_bcc_repeats(self):
        lat = Lattice.bcc(a=2.0, species="Fe")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 16
        assert all(s == "Fe" for s in structure.atoms["element"])

    def test_fcc_repeats(self):
        lat = Lattice.fcc(a=3.52, species="Ni")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 32
        assert all(s == "Ni" for s in structure.atoms["element"])

    def test_rocksalt_repeats(self):
        lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 64
        na = sum(1 for s in structure.atoms["element"] if s == "Na")
        cl = sum(1 for s in structure.atoms["element"] if s == "Cl")
        assert na == 32
        assert cl == 32

    def test_empty_basis(self):
        lat = Lattice(cell=np.eye(3))
        structure = lat.build(repeats=(2, 2, 2))

        assert isinstance(structure, Atomistic)
        assert len(list(structure.atoms)) == 0

    def test_super_cell_box(self):
        lat = Lattice.sc(a=3.0, species="Cu")

        box = lat.supercell((2, 2, 2))
        assert isinstance(box, Box)
        assert np.allclose(box.h, 6.0 * np.eye(3))

    def test_super_cell_box_holds_lattice_vectors_as_columns(self):
        a1, a2, a3 = [2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]
        lat = Lattice.from_vectors(a1, a2, a3)

        box = lat.supercell((1, 2, 1))
        np.testing.assert_allclose(box.h[:, 0], a1)
        np.testing.assert_allclose(box.h[:, 1], 2.0 * np.asarray(a2))
        np.testing.assert_allclose(box.h[:, 2], a3)

    def test_the_cell_is_never_written_onto_the_structure(self):
        """A structure is topology and chemistry; the cell lives on ``frame.box``."""
        lat = Lattice.sc(a=3.0, species="Cu")
        structure = lat.build(repeats=(2, 2, 2))

        assert "box" not in structure.props

    def test_positions(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        structure = lat.build(repeats=(2, 2, 2))

        positions = structure.atoms["x", "y", "z"]
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
        structure = lat.build(Cuboid.cube(3.0))

        # cells inferred = ceil(3/2)=2 along each axis → 8 atoms generated,
        # all inside the [0,3]³ region (positions ∈ {0, 2}).
        assert len(list(structure.atoms)) == 8

    def test_box_region_clips_extra_atoms(self):
        lat = Lattice.sc(a=2.0, species="Cu")
        # Force a 3-cell tile but clip to a 3 Å box: corner atoms at x=4 etc.
        # are filtered out.
        structure = lat.build(Cuboid.cube(3.0), repeats=(3, 3, 3))

        assert len(list(structure.atoms)) == 8

    def test_sphere_region(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        structure = lat.build(
            Sphere([1.5, 1.5, 1.5], 1.5),
            repeats=(4, 4, 4),
        )

        # Every atom must lie within the sphere.
        positions = structure.atoms["x", "y", "z"]
        center = np.array([1.5, 1.5, 1.5])
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.all(distances <= 1.5 + 1e-9)
        assert len(positions) > 0

    def test_combined_regions(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        cube = Cuboid.cube(3.0)
        sphere = Sphere([1.5, 1.5, 1.5], 1.5)
        # Intersection: atoms in both.
        structure = lat.build(cube & sphere, repeats=(4, 4, 4))

        positions = structure.atoms["x", "y", "z"]
        center = np.array([1.5, 1.5, 1.5])
        in_sphere = np.linalg.norm(positions - center, axis=1) <= 1.5 + 1e-9
        in_box = np.all((positions >= 0) & (positions <= 3.0), axis=1)
        assert np.all(in_sphere & in_box)

    def test_requires_region_or_repeats(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="region.*repeats"):
            lat.build()

    def test_rejects_non_positive_repeats(self):
        lat = Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="repeats must be positive"):
            lat.build(repeats=(0, 1, 1))
