import numpy as np
import molpy as mp

class TestSimulator:

    def test_full_simulation(self):

        n_atoms = 10
        frame = mp.Frame(n_atoms = n_atoms)
        frame.box = mp.Box([5, 5, 5])
        frame.positions = np.random.rand(n_atoms, 3)
        frame.mass = np.ones((n_atoms, 1))
        frame.momenta = np.zeros((n_atoms, 3))
        frame.forces = np.zeros((n_atoms, 3))
        frame.energy = np.zeros((n_atoms, 1))

        simulator = mp.md.Simulator(
            frame,
            potential=mp.potential.pair.LJ126(sigma=1.0, epsilon=1.0, cutoff=2.5),
            fixes=[mp.md.fix.Langevin(1.0, 1.0)],
            neighborlist=mp.NeighborList(cutoff=2.4),
            integrator=mp.md.integrator.VelocityVerlet(0.001),
        )
        simulator.run(10)