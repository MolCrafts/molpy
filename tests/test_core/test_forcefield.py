import pytest
import numpy as np
import numpy.testing as npt
import molpy as mp


class TestForceField:

    @pytest.fixture(scope="class", name="ff")
    def init_forcefield(self):

        ff = mp.ForceField()
        return ff

    def test_bond(self, ff):

        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_bondtype("O-H", 0, 1, r0=1.0)

        npt.assert_equal(
            bondstyle.get_bondtype_params("r0"), np.array([[0.0, 1.0], [1.0, 0.0]])
        )

    def test_pair(self, ff):

        pairstyle = ff.def_pairstyle("lj/cut/coul/cut", global_cutoff=10.0, mix="arithmetic")
        pairstyle.def_pairtype("O-O", 0, 0, epsilon=0.1553, sigma=3.1506)
        pairstyle.def_pairtype("O-H", 0, 1, epsilon=0.0, sigma=1.0)
        pairstyle.def_pairtype("H-H", 1, 1, epsilon=0.0, sigma=1.0)

        npt.assert_allclose(
            pairstyle.get_pairtype_params("epsilon"),
            np.array([[0.1553, 0.0], [0.0, 0.0]]),
        )
        npt.assert_allclose(
            pairstyle.get_pairtype_params("sigma"),
            np.array([[3.1506, 1.0], [1.0, 1.0]]),
        )
