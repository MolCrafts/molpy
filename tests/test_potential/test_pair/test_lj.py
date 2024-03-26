import numpy as np
import numpy.testing as npt
from molpy.potential.pair.lj import LJ126


class TestLJ126:

    def test_energy_forces(self):

        n_types = 3
        epsilon = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        sigma = np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])

        lj126 = LJ126(epsilon, sigma, cutoff=2.0)

        R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        atomtypes = np.array([0, 1, 2])
        pairs = np.array([[0, 1], [0, 2], [1, 2]])
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        energy = lj126.energy(R, atomtypes, idx_i, idx_j)
        forces = lj126.forces(R, atomtypes, idx_i, idx_j)


        rij = R[idx_j] - R[idx_i]
        dij = np.linalg.norm(rij, axis=-1)
        r01 = rij[0]
        d01 = dij[0]
        power_6 = np.power(sigma[0, 1] / d01, 6)
        power_12 = np.square(power_6)
        e = 4 * epsilon[0, 1] * (power_12 - power_6)
        f = 24 * epsilon[0, 1] * (2 * power_12 - power_6) / d01**2 * r01
        npt.assert_allclose(energy[0], e)
        npt.assert_allclose(forces[0], f)
