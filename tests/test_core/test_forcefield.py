import pytest
import numpy as np
import numpy.testing as npt
import molpy as mp

from molpy.core.forcefield import Style, Type

class TestStyle:

    def test_init_with_str(self):

        style = Style("lj/cut/coul/cut", mixing="arithmetic")
        assert style.style == "lj/cut/coul/cut"
        assert style.mixing == "arithmetic"

    def test_init_with_potential(self):

        assert Style(mp.Potential) == {}

class TestForceField:

    @pytest.fixture(scope="class", name="ff")
    def init_forcefield(self):

        ff = mp.ForceField()
        return ff
    
    def test_atom(self, ff:mp.ForceField):

        atomstyle = ff.def_atomstyle("atomic")
        atomstyle.def_atomtype("O", 0, mass=15.9994)
        atomstyle.def_atomtype("H", 1, mass=1.00794)

        assert atomstyle.n_types == 2

    def test_bond(self, ff:mp.ForceField):

        bondstyle = ff.def_bondstyle(mp.potential.bond.Harmonic, )
        bondstyle.def_bondtype("O-H", 0, 1, r0=1.012, k=1059.162)
        params = bondstyle.get_params("r0", format="numpy")
        npt.assert_allclose(
            params, 
            np.array([
                [0, 1.012],
                [1.012, 0]
            ])
        )
        assert bondstyle.n_types == 2  # O-H, H-O


    def test_customized_bond(self, ff:mp.ForceField):

        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_bondtype("O-H", 0, 1, r0=1.0)

        npt.assert_equal(
            bondstyle.get_params("r0"), np.array([[0.0, 1.0], [1.0, 0.0]])
        )

    # def test_pair(self, ff:mp.ForceField):

    #     pairstyle = ff.def_pairstyle("lj/cut/coul/cut", global_cutoff=10.0, mixing="arithmetic")
    #     pairstyle.def_pairtype("O-O", 0, 0, epsilon=0.1553, sigma=3.1506)
    #     pairstyle.def_pairtype("O-H", 0, 1, epsilon=0.0, sigma=1.0)
    #     pairstyle.def_pairtype("H-H", 1, 1, epsilon=0.0, sigma=1.0)

    #     npt.assert_allclose(
    #         pairstyle.get_pairtype_params("epsilon"),
    #         np.array([[0.1553, 0.0], [0.0, 0.0]]),
    #     )
    #     npt.assert_allclose(
    #         pairstyle.get_pairtype_params("sigma"),
    #         np.array([[3.1506, 1.0], [1.0, 1.0]]),
    #     )