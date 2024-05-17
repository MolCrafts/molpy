import molpy as mp
import numpy as np

class TestBondHarmonic:

    def test_energy(self):

        harmonic = mp.potential.bond.Harmonic()
        xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        idx_i = np.array([0])
        idx_j = np.array([1])
        type_i = np.array([0])
        type_j = np.array([0])
        k = np.array([[1.0]])
        r0 = np.array([[1.0]])
        e = harmonic.energy(xyz, idx_i, idx_j, type_i, type_j, k, r0)
        assert e == 0.5