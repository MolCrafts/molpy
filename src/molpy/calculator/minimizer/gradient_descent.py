import numpy as np
from .minimizer import Minimizer
from molpy import Alias

class SGD(Minimizer):
    def __init__(self, learning_rate=0.01, report_config={'rate': 10}, dump_config={}):
        self.learning_rate = learning_rate
        super().__init__(report_config, dump_config)

    def minimize(self, frame, potentials, nblist, max_iter=1000, tol=1e-3):

        self.before_minimize(frame, potentials)
        nblist(frame)

        start_status = potentials(frame)
        energy = start_status[Alias.energy]
        forces = start_status.atoms[Alias.forces]
        init_energy = energy
        init_forces = forces
        step = 0
        while True:
            direction = -forces
            # frame.atoms.positions += self.learning_rate * (direction / np.linalg.norm(direction, axis=-1, keepdims=True))
            frame.atoms.positions -= self.learning_rate * direction
            status = potentials(frame)
            energy = status[Alias.energy]
            forces = status.atoms[Alias.forces]
            if step % self.report_rate == 0:
                print(f"Step: {step}, Energy: {energy}")

            if step % self.dump_rate == 0:
                self.dump(frame)

            if step >= max_iter or abs((init_energy - energy)/init_energy) < tol:
                break

            nblist(frame)
            step += 1

        return frame