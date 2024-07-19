import pytest
import numpy as np
import numpy.testing as npt
import molpy as mp

from molpy.core.forcefield import Style, Type


class TestStyle:

    def test_init_with_str(self):

        style = Style("lj/cut/coul/cut", mixing="arithmetic")
        assert style.name == "lj/cut/coul/cut"
        assert style.calculator is None
        assert style.mixing == "arithmetic"

    def test_init_with_potential(self):

        style = Style(mp.potential.bond.Harmonic)

        assert style.name == "harmonic"


class TestForceField:

    @pytest.fixture(scope="class", name="ff")
    def init_forcefield(self):

        ff = mp.ForceField()
        return ff

    def test_atom(self, ff: mp.ForceField):

        atomstyle = ff.def_atomstyle("atomic")
        atomstyle.def_atomtype("O", 0, mass=15.9994)
        atomstyle.def_atomtype("H", 1, mass=1.00794)

        assert atomstyle.n_types == 2

    def test_bond(self, ff: mp.ForceField):

        bondstyle = ff.def_bondstyle(
            mp.potential.bond.Harmonic,
        )
        bondstyle.def_bondtype(0, 1, r0=1.012, k=1059.162, name="O-H")
        params = bondstyle.get_param("r0")
        npt.assert_allclose(params, np.array([[0, 1.012], [1.012, 0]]))
        assert bondstyle.n_types == 1  # O-H, H-O

    def test_angle(self, ff: mp.ForceField):

        anglestyle = ff.def_anglestyle(mp.potential.angle.Harmonic)
        anglestyle.def_angletype("H-O-H", 1, 0, 1, theta0=104.52, k=75.90)

        n_atomtypes = ff.n_atomtypes
        n_angletype = anglestyle.n_types
        assert n_angletype == 1, ValueError(f"Expected 2 atom types, got {n_angletype}")
        theta0 = anglestyle.get_param("theta0")
        assert theta0.shape == (n_atomtypes, n_atomtypes, n_atomtypes)

        expected_theta0 = np.array([[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        expected_theta0[1, 0, 1] = 104.52

        npt.assert_equal(
            theta0,
            expected_theta0
        )
