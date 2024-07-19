import molpy as mp
import numpy as np

class TestBondHarmonic:

    def test_energy(self):

        k = np.array([[1.0]])
        r = np.array([[1.5]])
        r0 = np.array([[1.0]])
        e = mp.potential.bond.Harmonic.E(k, r, r0)
        assert e == 0.125

    def test_forces(self):

        k = np.array([[1.0]])
        r = np.array([[1.5]])
        r0 = np.array([[1.0]])
        f = mp.potential.bond.Harmonic.F(k, r, r0)
        assert f == -0.5