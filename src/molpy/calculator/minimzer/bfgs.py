
import numpy as np
import molpy as mp

class Minimizer:
    pass

class BFGS(Minimizer):

    def __init__(self, frame:mp.Frame, delta:float, alpha: float):
        self.frame = frame
        self.delta = delta
        self.alpha = alpha

    def reset(self):

        self.B0 = np.eye(3 * self.frame.n_atoms) * self.alpha
        self.B = None
        self.pos0 = None
        self.forces0 = None

    def step(self, forces=None):
        
        if forces is None:
            forces = self.frame.forces

        pos = self.frame.positions
        dpos, step_lengths = self._get_gradient(pos, forces)
        dpos = self._modify_step(dpos, step_lengths)
        self.frame.positions = pos + dpos
        
    def _get_gradient(self, pos, forces):
        forces = forces.reshape(-1)
        self._update_B(pos.flat, forces, self.pos0, self.forces0)
        omega, V = np.linalg.eigh(self.B)

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        step_lengths = (dpos**2).sum(1)**0.5
        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()
        return dpos, step_lengths
    
    def _modify_step(self, dpos, step_lengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(step_lengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            dpos *= scale
        return dpos
    
    def _update_B(self, pos, forces, pos0, forces0):
        if self.B is None:
            self.B = self.B0
            return
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dforces = forces - forces0
        a = np.dot(dpos, dforces)
        dg = np.dot(self.B, dpos)
        b = np.dot(dpos, dg)
        self.B -= np.outer(dforces, dforces) / a + np.outer(dg, dg) / b

    def is_converged(self, etol, ftol):
        pass

    def minimize(self, max_steps:int, fmax:float):

        is_converged = self.is_converged()

        for step in range(max_steps):
            
            self.step()
            is_converged = self.is_converged()

            # traj and log

            if is_converged:
                break