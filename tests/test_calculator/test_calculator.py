import numpy as np
import molpy as mp

class TestMDCalculator:

    def test_langevin(self):

        n_atoms = 10
        box_size = 5
        frame = mp.Frame()
        frame.box = mp.Box.cube(box_size)
        frame.positions = np.random.rand(n_atoms, 3) * 5
        frame.mass = np.ones((n_atoms, 1))
        frame.momenta = np.zeros((n_atoms, 3))
        frame.forces = np.zeros((n_atoms, 3))
        frame.energy = np.zeros((n_atoms, 1))
        frame.types = np.ones(n_atoms, dtype=int)

        simulator = mp.md.Calculator(
            frame,
            potential=mp.potential.pair.LJ126(sigma=1.0, epsilon=1.0, cutoff=2.5),
            fixes=[mp.md.fix.Langevin(1.0, 1.0)],
            neighborlist=mp.NeighborList(cutoff=2.4),
            integrator=mp.md.integrator.VelocityVerlet(0.001),
            dump_config={"n_dump": 1, "file": "md.lammpstrj"},
        )
        simulator.run(10)

    def test_nve(self):

        box_size = 10
        frame = mp.Frame()
        frame.box = mp.Box.cube(box_size)
        frame.positions = np.array([[0, 0, 0], [1.1224, 0, 0]])
        frame.mass = np.ones((2, 1))
        frame.types = np.ones(2, dtype=int)

        simulator = mp.md.Calculator(
            frame,
            potential=mp.potential.pair.LJ126(sigma=1.0, epsilon=1.0, cutoff=2.5),
            neighborlist=mp.NeighborList(cutoff=2.5),
            integrator=mp.md.integrator.VelocityVerlet(1.0),
            dump_config={"n_dump": 1, "file": "md.lammpstrj"},
        )
        simulator.run(10)
        assert False