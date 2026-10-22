from __future__ import annotations


import dataclasses

import numpy as np
import pytest
import molpy as mp


class TestSite:
    def test_site_creation(self):
        site = mp.builder.Site(label="A", species="Ni", fractional=(0.0, 0.0, 0.0))
        assert site.label == "A"
        assert site.species == "Ni"
        assert site.fractional == (0.0, 0.0, 0.0)
        assert site.charge == 0.0
        assert site.attrs is None

    def test_site_with_charge(self):
        site = mp.builder.Site(
            label="Na", species="Na", fractional=(0.0, 0.0, 0.0), charge=1.0
        )
        assert site.charge == 1.0

    def test_site_with_attrs(self):
        attrs = {"tag": "corner"}
        site = mp.builder.Site(
            label="A", species="C", fractional=(0.0, 0.0, 0.0), attrs=attrs
        )
        assert site.attrs == attrs

    def test_site_attrs_are_copied_to_built_atoms(self):
        site = mp.builder.Site(
            label="A",
            species="C",
            fractional=(0.0, 0.0, 0.0),
            attrs={"domain": "wall"},
        )
        structure = mp.builder.Lattice(
            mp.Box(mp.core.Cell.lengths([1.0] * 3)), [site]
        ).build(repeats=(1, 1, 1))
        assert structure.atoms[0].get("domain") == "wall"

    def test_site_is_frozen(self):
        site = mp.builder.Site(label="A", species="C", fractional=(0.0, 0.0, 0.0))
        with pytest.raises(dataclasses.FrozenInstanceError):
            site.label = "B"  # type: ignore[misc]


class TestLattice:
    def test_lattice_holds_a_box(self):
        box = mp.Box(mp.core.Cell.lengths([1.0] * 3))
        lattice = mp.builder.Lattice(box, basis=[])

        assert lattice.box is box
        assert lattice.basis == ()

    def test_lattice_rejects_a_matrix(self):
        with pytest.raises(TypeError, match="Box"):
            mp.builder.Lattice(np.eye(3))  # type: ignore[arg-type]

    def test_lattice_rejects_a_box_without_a_cell(self):
        with pytest.raises(ValueError, match="cell"):
            mp.builder.Lattice(mp.Box(mp.core.Cell.undefined()))

    def test_lattice_vectors_are_the_box_columns(self):
        h = np.array([[3.0, 1.0, 0.5], [0.0, 4.0, 0.2], [0.0, 0.0, 5.0]])
        lattice = mp.builder.Lattice(mp.Box(mp.core.Cell.matrix(h)))

        np.testing.assert_allclose(lattice.a1, h[:, 0])
        np.testing.assert_allclose(lattice.a2, h[:, 1])
        np.testing.assert_allclose(lattice.a3, h[:, 2])

    def test_from_vectors(self):
        a1 = np.array([3.0, 0.0, 0.0])
        a2 = np.array([1.0, 3.0, 0.0])
        a3 = np.array([0.5, 0.5, 3.0])
        lattice = mp.builder.Lattice.from_vectors(a1, a2, a3, basis=[])

        assert isinstance(lattice.box, mp.Box)
        np.testing.assert_allclose(lattice.box.matrix, np.column_stack([a1, a2, a3]))
        np.testing.assert_allclose(lattice.a2, a2)

    def test_with_site_is_immutable(self):
        base = mp.builder.Lattice(mp.Box(mp.core.Cell.lengths([1.0] * 3)))
        site = mp.builder.Site(label="A", species="C", fractional=(0.0, 0.0, 0.0))
        extended = base.with_site(site)

        assert base.basis == ()
        assert extended.basis == (site,)
        assert extended is not base
        assert extended.box is base.box

    def test_box_to_cart_single(self):
        lattice = mp.builder.Lattice(
            mp.Box(mp.core.Cell.lengths(np.array([3.0, 4.0, 5.0])))
        )

        cart = lattice.box.to_cart(np.array([[0.5, 0.5, 0.5]]))[0]
        assert np.allclose(cart, [1.5, 2.0, 2.5])

    def test_box_to_cart_multiple(self):
        lattice = mp.builder.Lattice(mp.Box(mp.core.Cell.lengths([2.0] * 3)))

        frac = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.5]]
        )
        cart = lattice.box.to_cart(frac)
        expected = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]]
        )
        assert np.allclose(cart, expected)

    def test_box_to_frac_inverts_to_cart_and_columns_are_lattice_vectors(self):
        a1, a2, a3 = [2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 3.0]
        lattice = mp.builder.Lattice.from_vectors(a1, a2, a3)

        frac = np.array([[0.1, 0.2, 0.3], [0.7, 0.4, 0.9]])
        cart = lattice.box.to_cart(frac)
        recovered = lattice.box.to_frac(cart)
        assert np.allclose(recovered, frac)
        # cart = u a1 + v a2 + w a3.
        assert np.allclose(cart, frac @ np.array([a1, a2, a3]))

    def test_sc(self):
        lat = mp.builder.Lattice.sc(a=2.0, species="Cu")

        assert np.allclose(lat.box.matrix, 2.0 * np.eye(3))
        assert len(lat.basis) == 1
        assert lat.basis[0].species == "Cu"
        assert lat.basis[0].fractional == (0.0, 0.0, 0.0)

    def test_bcc(self):
        lat = mp.builder.Lattice.bcc(a=3.0, species="Fe")

        assert np.allclose(lat.box.matrix, 3.0 * np.eye(3))
        assert len(lat.basis) == 2

        fracs = [site.fractional for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.5) in fracs
        assert all(site.species == "Fe" for site in lat.basis)

    def test_fcc(self):
        lat = mp.builder.Lattice.fcc(a=3.52, species="Ni")

        assert np.allclose(lat.box.matrix, 3.52 * np.eye(3))
        assert len(lat.basis) == 4
        fracs = [site.fractional for site in lat.basis]
        assert (0.0, 0.0, 0.0) in fracs
        assert (0.5, 0.5, 0.0) in fracs
        assert (0.5, 0.0, 0.5) in fracs
        assert (0.0, 0.5, 0.5) in fracs
        assert all(site.species == "Ni" for site in lat.basis)

    def test_rocksalt(self):
        lat = mp.builder.Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")

        assert np.allclose(lat.box.matrix, 5.64 * np.eye(3))
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

    def test_from_spacegroup_takes_a_box(self):
        group = mp.builder.SpaceGroup.from_triplets(["x,y,z", "-x,-y,-z"])
        box = mp.Box(mp.core.Cell.lengths([4.0] * 3))
        lat = mp.builder.Lattice.from_spacegroup(
            box, [mp.builder.Site("C1", "C", (0.1, 0.2, 0.3))], group
        )

        assert lat.box is box
        assert [s.label for s in lat.basis] == ["C1_0", "C1_1"]
        np.testing.assert_allclose(lat.basis[1].fractional, [0.9, 0.8, 0.7])


class TestBuildCrystalRepeats:
    def test_sc_repeats(self):
        lat = mp.builder.Lattice.sc(a=2.0, species="Cu")
        structure = lat.build(repeats=(2, 2, 2))

        assert isinstance(structure, mp.Atomistic)
        assert len(list(structure.atoms)) == 8
        assert all(s == "Cu" for s in structure.atoms["element"])

    def test_bcc_repeats(self):
        lat = mp.builder.Lattice.bcc(a=2.0, species="Fe")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 16
        assert all(s == "Fe" for s in structure.atoms["element"])

    def test_fcc_repeats(self):
        lat = mp.builder.Lattice.fcc(a=3.52, species="Ni")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 32
        assert all(s == "Ni" for s in structure.atoms["element"])

    def test_rocksalt_repeats(self):
        lat = mp.builder.Lattice.rocksalt(a=5.64, species_a="Na", species_b="Cl")
        structure = lat.build(repeats=(2, 2, 2))

        assert len(list(structure.atoms)) == 64
        na = sum(1 for s in structure.atoms["element"] if s == "Na")
        cl = sum(1 for s in structure.atoms["element"] if s == "Cl")
        assert na == 32
        assert cl == 32

    def test_empty_basis(self):
        lat = mp.builder.Lattice(mp.Box(mp.core.Cell.lengths([1.0] * 3)))
        structure = lat.build(repeats=(2, 2, 2))

        assert isinstance(structure, mp.Atomistic)
        assert len(list(structure.atoms)) == 0

    def test_super_cell_box(self):
        lat = mp.builder.Lattice.sc(a=3.0, species="Cu")

        box = lat.supercell((2, 2, 2))
        assert isinstance(box, mp.Box)
        assert np.allclose(box.matrix, 6.0 * np.eye(3))

    def test_super_cell_box_holds_lattice_vectors_as_columns(self):
        a1, a2, a3 = [2.0, 0.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 4.0]
        lat = mp.builder.Lattice.from_vectors(a1, a2, a3)

        box = lat.supercell((1, 2, 1))
        np.testing.assert_allclose(box.matrix[:, 0], a1)
        np.testing.assert_allclose(box.matrix[:, 1], 2.0 * np.asarray(a2))
        np.testing.assert_allclose(box.matrix[:, 2], a3)

    def test_super_cell_keeps_origin_and_periodicity(self):
        box = mp.Box(
            mp.core.Cell.matrix(np.diag([1.0, 2.0, 3.0])),
            origin=np.array([1.0, -1.0, 0.5]),
            pbc=np.array([True, False, True]),
        )
        supercell = mp.builder.Lattice(box).supercell((2, 1, 3))

        np.testing.assert_allclose(supercell.matrix, np.diag([2.0, 2.0, 9.0]))
        np.testing.assert_allclose(supercell.origin, box.origin)
        assert supercell.pbc.tolist() == [True, False, True]

    def test_the_cell_is_never_written_onto_the_structure(self):
        """A structure is topology and chemistry; the cell lives on ``frame.box``."""
        lat = mp.builder.Lattice.sc(a=3.0, species="Cu")
        structure = lat.build(repeats=(2, 2, 2))

        assert "box" not in structure.props

    def test_positions(self):
        lat = mp.builder.Lattice.sc(a=2.0, species="Cu")
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
        lat = mp.builder.Lattice.sc(a=2.0, species="Cu")
        structure = lat.build(mp.core.Cuboid((0, 0, 0), (3.0, 3.0, 3.0)))

        # cells inferred = ceil(3/2)=2 along each axis → 8 atoms generated,
        # all inside the [0,3]³ region (positions ∈ {0, 2}).
        assert len(list(structure.atoms)) == 8

    def test_box_region_clips_extra_atoms(self):
        lat = mp.builder.Lattice.sc(a=2.0, species="Cu")
        # Force a 3-cell tile but clip to a 3 Å box: corner atoms at x=4 etc.
        # are filtered out.
        structure = lat.build(
            mp.core.Cuboid((0, 0, 0), (3.0, 3.0, 3.0)), repeats=(3, 3, 3)
        )

        assert len(list(structure.atoms)) == 8

    def test_sphere_region(self):
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        structure = lat.build(
            mp.core.Sphere([1.5, 1.5, 1.5], 1.5),
            repeats=(4, 4, 4),
        )

        # Every atom must lie within the sphere.
        positions = structure.atoms["x", "y", "z"]
        center = np.array([1.5, 1.5, 1.5])
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.all(distances <= 1.5 + 1e-9)
        assert len(positions) > 0

    def test_combined_regions(self):
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        cube = mp.core.Cuboid((0, 0, 0), (3.0, 3.0, 3.0))
        sphere = mp.core.Sphere([1.5, 1.5, 1.5], 1.5)
        # Intersection: atoms in both.
        structure = lat.build(cube & sphere, repeats=(4, 4, 4))

        positions = structure.atoms["x", "y", "z"]
        center = np.array([1.5, 1.5, 1.5])
        in_sphere = np.linalg.norm(positions - center, axis=1) <= 1.5 + 1e-9
        in_box = np.all((positions >= 0) & (positions <= 3.0), axis=1)
        assert np.all(in_sphere & in_box)

    def test_sphere_at_origin_is_fully_covered(self):
        """Inferred tiling spans negative cells too, not just the +++ octant."""
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        structure = lat.build(mp.core.Sphere([0.0, 0.0, 0.0], 2.5))

        grid = np.stack(
            np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        expected = grid[np.linalg.norm(grid, axis=1) <= 2.5]
        positions = structure.atoms["x", "y", "z"]
        assert len(positions) == len(expected)
        assert positions[:, 0].min() == pytest.approx(-2.0)
        assert positions[:, 0].max() == pytest.approx(2.0)

    def test_offset_region_on_a_sheared_cell_is_fully_covered(self):
        a1, a2, a3 = [2.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.5, 0.5, 2.0]
        lat = mp.builder.Lattice.from_vectors(
            a1, a2, a3, basis=[mp.builder.Site("A", "C", (0.0, 0.0, 0.0))]
        )
        region = mp.core.Sphere([-3.0, 4.0, -1.0], 5.0)
        structure = lat.build(region)

        ijk = np.stack(
            np.meshgrid(*[np.arange(-10, 11)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        points = ijk @ np.array([a1, a2, a3])
        expected = region.contains(points).sum()
        assert len(list(structure.atoms)) == expected

    def test_unbounded_region_needs_repeats(self):
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="unbounded"):
            lat.build(mp.core.HalfSpace([0.0, 0.0, 1.0], [0.0, 0.0, 0.0]))

    def test_requires_region_or_repeats(self):
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="region.*repeats"):
            lat.build()

    def test_rejects_non_positive_repeats(self):
        lat = mp.builder.Lattice.sc(a=1.0, species="Cu")
        with pytest.raises(ValueError, match="repeats must be three positive integers"):
            lat.build(repeats=(0, 1, 1))
