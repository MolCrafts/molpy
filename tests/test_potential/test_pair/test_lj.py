import numpy as np
import numpy.testing as npt
from molpy.potential.pair.lj import LJ126


class TestLJ126:

    def test_energy(self):
        eps = np.array([[1.0]])
        sig = np.array([[1.0]])
        r = np.array([[1.0]])
        e = LJ126.E(r, eps, sig)
        npt.assert_allclose(e, -1.0)

    def test_forces(self):
        eps = np.array([[1.0]])
        sig = np.array([[1.0]])
        r = np.array([[1.0]])
        f = LJ126.F(r, eps, sig)
        npt.assert_allclose(f, 0.0)

    def test_forward(self):
        pass