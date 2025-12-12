from __future__ import annotations

import numpy as np
import pytest

from molpy import Atomistic
from molpy.builder.crystal import BlockRegion, CrystalBuilder, Lattice, Site


class TestSite:
    def test_site_creation(self):
        site = Site(label="A", species="Ni", frac=(0.0, 0.0, 0.0))
        assert site.label == "A"
        assert site.species == "Ni"
        assert site.frac == (0.0, 0.0, 0.0)
        assert site.charge == 0.0
        assert site.meta is None

    def test_site_with_charge(self):
        site = Site(label="Na", species="Na", frac=(0.0, 0.0, 0.0), charge=1.0)
        assert site.charge == 1.0

    def test_site_with_metadata(self):
        meta = {"tag": "corner"}
        site = Site(label="A", species="C", frac=(0.0, 0.0, 0.0), meta=meta)
        assert site.meta == meta


class TestLattice:
    def test_lattice_creation(self):
        a1 = np.array([1.0, 0.0, 0.0])
        a2 = np.array([0.0, 1.0, 0.0])
        a3 = np.array([0.0, 0.0, 1.0])
        lattice = Lattice(a1=a1, a2=a2, a3=a3, basis=[])

        assert np.allclose(lattice.a1, a1)
        assert np.allclose(lattice.a2, a2)
        assert np.allclose(lattice.a3, a3)
        assert len(lattice.basis) == 0

    def test_lattice_cell_property(self):
        a1 = np.array([3.0, 0.0, 0.0])
        a2 = np.array([0.0, 3.0, 0.0])
        a3 = np.array([0.0, 0.0, 3.0])
        lattice = Lattice(a1=a1, a2=a2, a3=a3, basis=[])

        cell = lattice.cell
        assert cell.shape == (3, 3)
        assert np.allclose(cell[0], a1)
        assert np.allclose(cell[1], a2)
        assert np.allclose(cell[2], a3)

    def test_add_site(self):
        lattice = Lattice(
            a1=np.array([1.0, 0.0, 0.0]),
            a2=np.array([0.0, 1.0, 0.0]),
            a3=np.array([0.0, 0.0, 1.0]),
            basis=[],
        )
        site = Site(label="A", species="C", frac=(0.0, 0.0, 0.0))
        lattice.add_site(site)
        assert len(lattice.basis) == 1
        assert lattice.basis[0] == site

    def test_frac_to_cart_single(self):
        a1 = np.array([3.0, 0.0, 0.0])
        a2 = np.array([0.0, 4.0, 0.0])
        a3 = np.array([0.0, 0.0, 5.0])
        lattice = Lattice(a1=a1, a2=a2, a3=a3, basis=[])

        frac = np.array([0.5, 0.5, 0.5])
        cart = lattice.frac_to_cart(frac)
        expected = np.array([1.5, 2.0, 2.5])
        assert np.allclose(cart, expected)

    def test_frac_to_cart_multiple(self):
        a1 = np.array([2.0, 0.0, 0.0])
        a2 = np.array([0.0, 2.0, 0.0])
        a3 = np.array([0.0, 0.0, 2.0])
        lattice = Lattice(a1=a1, a2=a2, a3=a3, basis=[])

        frac = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]]
        )
        cart = lattice.frac_to_cart(frac)
        expected = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
        )
        assert np.allclose(cart, expected)

    def test_cubic_sc(self):
        lat = Lattice.cubic_sc(a=2.0, species="Cu")

        assert np.allclose(lat.a1, [2.0, 0.0, 0.0])
        assert np.allclose(lat.a2, [0.0, 2.0, 0.0])
        assert np.allclose(lat.a3, [0.0, 0.0, 2.0])

        assert len(lat.basis) == 1
        assert lat.basis[0].species == "Cu"
        assert lat.basis[0].frac == (0.0, 0.0, 0.0)

    def test_cubic_bcc(self):
        lat = Lattice.cubic_bcc(a=3.0, species="Fe")

        assert np.allclose(lat.a1, [3.0, 0.0, 0.0])
        assert len(lat.basis) == 2

        fracs = [site.frac for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.5) in fracs

        for site in lat.basis:
            assert site.species == "Fe"

    def test_cubic_fcc(self):
        lat = Lattice.cubic_fcc(a=3.52, species="Ni")

        assert len(lat.basis) == 4

        fracs = [site.frac for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.0) in fracs
        assert (0.5, 0.0, 0.5) in fracs
        assert (0.0, 0.5, 0.5) in fracs

        for site in lat.basis:
            assert site.species == "Ni"

    def test_rocksalt(self):
        lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")

        assert len(lat.basis) == 8

        na_count = sum(1 for site in lat.basis if site.species == "Na")
        cl_count = sum(1 for site in lat.basis if site.species == "Cl")
        assert na_count == 4
        assert cl_count == 4

        na_fracs = [site.frac for site in lat.basis if site.species == "Na"]
        assert (0.0, 0.0, 0.0) in na_fracs
        assert (0.5, 0.5, 0.0) in na_fracs

        cl_fracs = [site.frac for site in lat.basis if site.species == "Cl"]
        assert (0.5, 0.0, 0.0) in cl_fracs
        assert (0.0, 0.5, 0.0) in cl_fracs


class TestBlockRegion:
    def test_lattice_coord_system(self):
        region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="lattice")

        points = np.array(
            [
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
                [11.0, 5.0, 5.0],
                [-1.0, 5.0, 5.0],
            ]
        )

        mask = region.contains_mask(points)
        assert mask[0]
        assert mask[1]
        assert mask[2]
        assert not mask[3]
        assert not mask[4]

    def test_cartesian_coord_system(self):
        region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="cartesian")

        points = np.array(
            [
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 10.0],
                [11.0, 5.0, 5.0],
            ]
        )

        mask = region.contains_mask(points)
        assert mask[0]
        assert mask[1]
        assert mask[2]
        assert not mask[3]

    def test_partial_overlap(self):
        region = BlockRegion(-5, 5, -5, 5, -5, 5, coord_system="lattice")

        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [0.0, 6.0, 0.0],
                [-5.0, -5.0, -5.0],
            ]
        )

        mask = region.contains_mask(points)
        assert mask[0]
        assert mask[1]
        assert not mask[2]
        assert mask[3]


class TestCrystalBuilder:
    def test_simple_cubic_small(self):
        lat = Lattice.cubic_sc(a=2.0, species="Cu")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert isinstance(structure, Atomistic)

        assert len(list(structure.atoms)) == 8

        symbols = structure.symbols
        assert all(s == "Cu" for s in symbols)

    def test_simple_cubic_with_explicit_ranges(self):
        lat = Lattice.cubic_sc(a=3.0, species="Fe")
        region = BlockRegion(0, 5, 0, 5, 0, 5, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(
            region, i_range=range(0, 2), j_range=range(0, 2), k_range=range(0, 2)
        )

        assert len(list(structure.atoms)) == 8

        positions = structure.xyz
        assert positions.shape == (8, 3)

        assert np.allclose(positions[0], [0.0, 0.0, 0.0])

    def test_bcc_structure(self):
        lat = Lattice.cubic_bcc(a=2.0, species="Fe")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert len(list(structure.atoms)) == 16

        symbols = structure.symbols
        assert all(s == "Fe" for s in symbols)

    def test_fcc_structure(self):
        lat = Lattice.cubic_fcc(a=3.52, species="Ni")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert len(list(structure.atoms)) == 32

        symbols = structure.symbols
        assert all(s == "Ni" for s in symbols)

    def test_rocksalt_structure(self):
        lat = Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert len(list(structure.atoms)) == 64

        symbols = structure.symbols
        na_count = sum(1 for s in symbols if s == "Na")
        cl_count = sum(1 for s in symbols if s == "Cl")
        assert na_count == 32
        assert cl_count == 32

    def test_empty_basis(self):
        lat = Lattice(
            a1=np.array([1.0, 0.0, 0.0]),
            a2=np.array([0.0, 1.0, 0.0]),
            a3=np.array([0.0, 0.0, 1.0]),
            basis=[],
        )
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert len(list(structure.atoms)) == 0
        assert isinstance(structure, Atomistic)

    def test_cartesian_region_without_ranges_raises_error(self):
        lat = Lattice.cubic_sc(a=2.0, species="Cu")
        region = BlockRegion(0, 10, 0, 10, 0, 10, coord_system="cartesian")
        builder = CrystalBuilder(lat)

        with pytest.raises(ValueError, match=r"i_range.*j_range.*k_range"):
            builder.build_block(region)

    def test_cartesian_region_with_ranges(self):
        lat = Lattice.cubic_sc(a=2.0, species="Cu")
        region = BlockRegion(0, 3, 0, 3, 0, 3, coord_system="cartesian")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(
            region, i_range=range(0, 2), j_range=range(0, 2), k_range=range(0, 2)
        )

        assert len(list(structure.atoms)) == 8

    def test_positions_are_correct(self):
        lat = Lattice.cubic_sc(a=2.0, species="Cu")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)
        positions = structure.xyz

        expected_positions = np.array(
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

        positions_sorted = positions[
            np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        ]
        expected_sorted = expected_positions[
            np.lexsort(
                (
                    expected_positions[:, 2],
                    expected_positions[:, 1],
                    expected_positions[:, 0],
                )
            )
        ]

        assert np.allclose(positions_sorted, expected_sorted)

    def test_box_is_set_correctly(self):
        lat = Lattice.cubic_sc(a=3.0, species="Cu")
        region = BlockRegion(0, 2, 0, 2, 0, 2, coord_system="lattice")
        builder = CrystalBuilder(lat)

        structure = builder.build_block(region)

        assert "box" in structure
        from molpy import Box

        box = structure["box"]
        assert isinstance(box, Box)

        expected_cell = lat.cell
        assert np.allclose(box.matrix, expected_cell)
