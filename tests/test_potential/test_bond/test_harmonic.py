import numpy as np
import numpy.testing as npt
import molpy as mp

class TestBondHarmonic:

    def test_energy(self):

        r = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        bond_idx = np.array([[0, 1]])
        bond_types = np.array([0])

        harmonic = mp.potential.bond.Harmonic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        energy = harmonic.calc_energy(r, bond_idx, bond_types)
        npt.assert_allclose(energy, 0.5)

    def test_force(self):

        r = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        bond_idx = np.array([[0, 1]])
        bond_types = np.array([0])

        harmonic = mp.potential.bond.Harmonic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        force = harmonic.calc_force(r, bond_idx, bond_types)
        npt.assert_allclose(force, [[0.0, 0.0, 0.0]])

    def test_frame(self):

        frame = mp.Frame('atoms', 'bonds')
        frame['atoms']['x'] = np.array([0.0, 1.0])
        frame['atoms']['y'] = np.array([0.0, 0.0])
        frame['atoms']['z'] = np.array([0.0, 0.0])
        frame['bonds']['i'] = np.array([0])
        frame['bonds']['j'] = np.array([1])
        frame['bonds']['type'] = np.array([0])

        harmonic = mp.potential.bond.Harmonic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))

        energy = harmonic.calc_energy(frame)
        npt.assert_allclose(energy, 0.)