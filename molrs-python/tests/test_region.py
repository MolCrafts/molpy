import molrs
import numpy as np
import pytest


class TestSphere:
    def test_contains(self):
        s = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # center
                [1.0, 0.0, 0.0],  # inside
                [2.0, 0.0, 0.0],  # on surface
                [2.1, 0.0, 0.0],  # outside
            ],
            dtype=np.float64,
        )
        mask = s.contains(pts)
        assert mask[0] and mask[1] and mask[2]
        assert not mask[3]

    def test_bounds(self):
        s = molrs.core.Sphere(np.array([1.0, 2.0, 3.0], dtype=np.float64), 2.0)
        b = s.bounds()
        assert b.shape == (3, 2)
        np.testing.assert_allclose(b[:, 0], [-1.0, 0.0, 1.0], atol=1e-5)
        np.testing.assert_allclose(b[:, 1], [3.0, 4.0, 5.0], atol=1e-5)

    def test_bad_center_shape(self):
        with pytest.raises(ValueError, match="length 3"):
            molrs.core.Sphere(np.array([0.0, 0.0], dtype=np.float64), 1.0)

    def test_bad_points_shape(self):
        s = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        with pytest.raises(ValueError, match="N, 3"):
            s.contains(np.ones((3, 2), dtype=np.float64))

    def test_repr(self):
        s = molrs.core.Sphere(np.array([1.0, 2.0, 3.0], dtype=np.float64), 5.0)
        r = repr(s)
        assert "Sphere" in r
        assert "5.00" in r


class TestRegionAnd:
    def test_sphere_and_sphere(self):
        s1 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.core.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        intersection = s1 & s2

        pts = np.array(
            [
                [1.0, 0.0, 0.0],  # inside both
                [-2.5, 0.0, 0.0],  # inside s1 only
                [4.5, 0.0, 0.0],  # inside s2 only
            ],
            dtype=np.float64,
        )
        mask = intersection.contains(pts)
        assert mask[0] and not mask[1] and not mask[2]

    def test_sphere_and_cuboid(self):
        c = molrs.core.Cuboid(np.zeros(3), np.array([5.0, 5.0, 5.0]))
        s = molrs.core.Sphere(np.array([3.0, 0.0, 0.0], dtype=np.float64), 3.0)
        result = c & s
        assert isinstance(result, molrs.core.Region)


class TestRegionOr:
    def test_sphere_or_sphere(self):
        s1 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        s2 = molrs.core.Sphere(np.array([5.0, 0.0, 0.0], dtype=np.float64), 1.0)
        union = s1 | s2

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside s1
                [5.0, 0.0, 0.0],  # inside s2
                [2.5, 0.0, 0.0],  # outside both
            ],
            dtype=np.float64,
        )
        mask = union.contains(pts)
        assert mask[0] and mask[1] and not mask[2]


class TestRegionNot:
    def test_not_sphere(self):
        s = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        complement = ~s

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside sphere
                [3.0, 0.0, 0.0],  # outside sphere
            ],
            dtype=np.float64,
        )
        mask = complement.contains(pts)
        assert not mask[0] and mask[1]

    def test_not_cuboid(self):
        c = molrs.core.Cuboid(np.zeros(3), np.ones(3))
        result = ~c
        assert isinstance(result, molrs.core.Region)

    def test_complement_of_a_cube_refuses_the_atoms_on_its_faces(self):
        # A simple-cubic cell's atoms all sit on the cube's corners, edges
        # and faces: the boundary is the cube's, so `~cube` holds none of them.
        cube = molrs.core.Cuboid.cube(1.5)
        edge = np.array([0.0, 0.75, 1.5])
        grid = np.array([[x, y, z] for x in edge for y in edge for z in edge])
        pts = np.vstack([grid, [[3.0, 3.0, 3.0]]])
        block = molrs.core.Block({"x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2]})
        inside = cube.mask(block)
        outside = (~cube).mask(block)
        assert inside.tolist() == [True] * len(grid) + [False]
        assert outside.tolist() == [False] * len(grid) + [True]
        sphere = molrs.core.Sphere([0.0, 0.0, 0.0], 1.0)
        for region in (cube, sphere, cube & ~sphere, cube | sphere):
            assert not np.any(region.mask(block) & (~region).mask(block))
            assert np.all(region.mask(block) | (~region).mask(block))


class TestRegionChaining:
    def test_shell_via_and_not(self):
        outer = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 5.0)
        inner = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 2.0)
        shell = outer & (~inner)

        pts = np.array(
            [
                [0.0, 0.0, 0.0],  # inside inner
                [3.0, 0.0, 0.0],  # in shell
                [6.0, 0.0, 0.0],  # outside
            ],
            dtype=np.float64,
        )
        mask = shell.contains(pts)
        assert not mask[0] and mask[1] and not mask[2]

    def test_composed_bounds(self):
        s1 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.core.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        result = s1 & s2
        b = result.bounds()
        assert b.shape == (3, 2)

    def test_composed_repr(self):
        s1 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        s2 = molrs.core.Sphere(np.array([2.0, 0.0, 0.0], dtype=np.float64), 3.0)
        assert "composed" in repr(s1 & s2)

    def test_region_and_region(self):
        s1 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 5.0)
        s2 = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 3.0)
        r1 = s1 & s2
        r2 = ~s2
        r3 = r1 | r2  # composed & composed
        assert isinstance(r3, molrs.core.Region)

    def test_type_error_on_bad_operand(self):
        s = molrs.core.Sphere(np.array([0.0, 0.0, 0.0], dtype=np.float64), 1.0)
        with pytest.raises(TypeError):
            s & "not_a_region"


class TestDistance:
    def test_bad_points_shape(self):
        s = molrs.core.Sphere(np.zeros(3), 1.0)
        with pytest.raises(ValueError, match="N, 3"):
            s.distance(np.ones((3, 2), dtype=np.float64))


def _unit_cube_mesh() -> "molrs.core.TriMesh":
    v = np.array(
        [[x, y, z] for z in (0.0, 1.0) for y in (0.0, 1.0) for x in (0.0, 1.0)],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 4, 6],
            [0, 6, 2],
            [1, 3, 7],
            [1, 7, 5],
            [0, 1, 5],
            [0, 5, 4],
            [2, 6, 7],
            [2, 7, 3],
            [0, 2, 3],
            [0, 3, 1],
            [4, 5, 7],
            [4, 7, 6],
        ],
        dtype=np.uint32,
    )
    return molrs.core.TriMesh(v, faces)


class TestShapes:
    def test_half_space_and_its_complement(self):
        below = molrs.core.HalfSpace(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 5.0]))
        pts = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 9.0]])
        np.testing.assert_allclose(below.distance(pts), [-3.0, 4.0])
        assert list(below.contains(pts)) == [True, False]
        assert list((~below).contains(pts)) == [False, True]
        np.testing.assert_allclose(below.normal(), [0.0, 0.0, 1.0])

    def test_cylinder(self):
        c = molrs.core.Cylinder(
            np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 3.0]), 2.0, 5.0
        )
        pts = np.array([[1.0, 1.0, 2.5], [3.0, 1.0, 2.5], [1.0, 1.0, 7.0]])
        np.testing.assert_allclose(c.distance(pts), [-2.0, 0.0, 2.0])
        with pytest.raises(ValueError):
            molrs.core.Cylinder(np.zeros(3), np.zeros(3), 1.0, 1.0)

    def test_ellipsoid(self):
        e = molrs.core.Ellipsoid(np.zeros(3), np.array([2.0, 3.0, 1.0]))
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.5]])
        np.testing.assert_allclose(e.distance(pts), [-1.0, 0.0, 1.5])
        with pytest.raises(ValueError, match="semi-axes"):
            molrs.core.Ellipsoid(np.zeros(3), np.array([1.0, 0.0, 1.0]))


class TestPolyhedron:
    def test_unit_cube(self):
        mesh = _unit_cube_mesh()
        assert mesh.is_watertight()
        assert (mesh.n_vertices, mesh.n_faces) == (8, 12)
        cube = molrs.core.Polyhedron(mesh)
        pts = np.array([[0.5, 0.5, 0.5], [2.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
        np.testing.assert_allclose(cube.distance(pts), [-0.5, 1.0, 0.0], atol=1e-9)
        assert list(cube.contains(pts)) == [True, False, True]
        np.testing.assert_allclose(cube.bounds()[:, 1], [1.0, 1.0, 1.0])
        assert cube.mesh().n_faces == 12

    def test_open_mesh_is_rejected(self):
        mesh = _unit_cube_mesh()
        open_mesh = molrs.core.TriMesh(mesh.vertices(), mesh.faces()[:-1])
        assert not open_mesh.is_watertight()
        with pytest.raises(ValueError, match="watertight"):
            molrs.core.Polyhedron(open_mesh)

    def test_bad_face_index_is_rejected(self):
        with pytest.raises(ValueError, match="vertex table"):
            molrs.core.TriMesh(np.zeros((3, 3)), np.array([[0, 1, 7]], dtype=np.uint32))


class TestSphereUnion:
    def test_free_union_and_its_void(self):
        centers = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        u = molrs.core.SphereUnion(centers, np.array([1.0, 2.0]))
        assert u.n_spheres == 2
        pts = np.array([[20.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        np.testing.assert_allclose(u.distance(pts), [8.0, 3.0, -0.5])
        void = ~u
        assert list(void.contains(pts)) == [True, True, False]
        np.testing.assert_allclose(void.distance(pts), [-8.0, -3.0, 0.5])

    def test_rejects(self):
        with pytest.raises(ValueError):
            molrs.core.SphereUnion(np.zeros((0, 3)), 1.0)
        with pytest.raises(ValueError):
            molrs.core.SphereUnion(np.zeros((2, 3)), np.array([1.0]))
        with pytest.raises(ValueError):
            molrs.core.SphereUnion(np.zeros((1, 3)), 0.0)


class TestRegionSelection:
    @staticmethod
    def _block():
        block = molrs.core.Block()
        block.insert("x", np.array([0.0, 5.0, 0.5]))
        block.insert("y", np.zeros(3))
        block.insert("z", np.zeros(3))
        block.insert("id", np.array([1, 2, 3], dtype=np.uint64))
        return block

    def test_mask_and_call_read_the_block_coordinates(self):
        sphere = molrs.core.Sphere([0.0, 0.0, 0.0], 1.0)
        block = self._block()
        assert list(sphere.mask(block)) == [True, False, True]
        assert list(sphere(block)["id"]) == [1, 3]
        shell = sphere & ~molrs.core.Sphere([0.0, 0.0, 0.0], 0.25)
        assert list(shell.mask(block)) == [False, False, True]

    def test_a_non_region_operand_defers_to_its_own_operator(self):
        class Selector:
            def __rand__(self, other):
                return ("and", other)

            def __ror__(self, other):
                return ("or", other)

        sphere = molrs.core.Sphere([0.0, 0.0, 0.0], 1.0)
        assert sphere & Selector() == ("and", sphere)
        assert sphere | Selector() == ("or", sphere)
        with pytest.raises(TypeError):
            sphere & 3

    def test_shape_parameters_and_cube(self):
        cube = molrs.core.Cuboid.cube(2.0, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(cube.origin, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(cube.lengths, [2.0, 2.0, 2.0])
        sphere = molrs.core.Sphere([1.0, 2.0, 3.0], 4.0)
        np.testing.assert_allclose(sphere.center, [1.0, 2.0, 3.0])
        assert sphere.radius == 4.0
