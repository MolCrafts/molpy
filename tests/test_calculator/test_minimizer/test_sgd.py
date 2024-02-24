import numpy as np
import molpy as mp

class TestGD:

    def test_sgd(self):

        n_atoms = 2
        box_size = 5
        frame = mp.Frame()
        frame.box = mp.Box.cube(box_size)
        # frame.atoms.positions = np.random.rand(n_atoms, 3) * 5
        frame.atoms.positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        frame.atoms.mass = np.ones((n_atoms, 1))
        frame.atoms.forces = np.zeros((n_atoms, 3))
        frame.atoms.types = np.ones(n_atoms, dtype=int)
        frame.energy = np.zeros((n_atoms, 1))

        minimizer = mp.minimizer.SGD(
            learning_rate=0.001,
            report_config={'rate': 1, },
            dump_config={'rate': 10, 'path': 'test.lammpstrj'},
        )
        nblist = mp.NeighborList(cutoff=2.5)
        potentials = mp.Potentials(mp.potential.pair.LJ126(sigma=1.0, epsilon=1.0, cutoff=2.5))
        minimizer.minimize(frame, potentials, nblist, max_iter=500)
