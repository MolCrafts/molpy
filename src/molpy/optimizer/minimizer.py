import numpy as np


class BaseOptimizer:

    def __init__(self, potential):

        self.potential = potential

        self.status = {}
        self.fixes = {}

    def fix(self, event, status):
        for fix in self.fixes.get(event, []):
            fix(status)

    def dump(self, frame):
        pass

    def run(self, step:int):
        
        for _ in self.irun(step):
            ...

    def irun(self, frame, step):

        self.initialize(frame)
        
        while self.status['nstep'] < step:

            self.fix('before_step', self.status)
            self.step(frame)
            self.fix('after_step', self.status)

            yield

    def initialize(self):
        pass

    def step(self):
        pass

class BFGS(BaseOptimizer):

    def __init__(self, potential, alpha:float=70.0, step_limit: float=1.0):
        super().__init__(potential=potential)
        self.alpha = alpha
        self.maxstep = step_limit

    def initialize(self, frame):

        n_atoms = len(frame['atoms'])

        # initial hessian
        self.H0 = np.eye(3 * n_atoms) * self.alpha

        self.H = None
        self.pos0 = None
        self.forces0 = None

    def step(self, frame):

        xyz = frame['atoms']['x', 'y', 'z']
        forces = self.potential.calc_force(frame)

        dpos, steplengths = self.prepare_step(xyz, forces)
        dpos = self.determine_step(dpos, steplengths)
        frame['atoms']['x', 'y', 'z'] = xyz + dpos

    def prepare_step(self, pos, forces):

        forces = forces.reshape(-1)
        self.update(pos.flat, forces, self.pos0, self.forces0)
        omega, V = np.linalg.eigh(self.H)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dpos**2).sum(1)**0.5
        self.pos0 = pos.flat.copy()
        self.forces0 = forces.copy()
        return dpos, steplengths

    
    def determine_step(self, dpos, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the direction.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            dpos *= scale
        return dpos

    def update(self, pos, forces, pos0, forces0):
        if self.H is None:
            self.H = self.H0
            return
        dpos = pos - pos0

        if np.abs(dpos).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        dforces = forces - forces0
        a = np.dot(dpos, dforces)
        dg = np.dot(self.H, dpos)
        b = np.dot(dpos, dg)
        self.H -= np.outer(dforces, dforces) / a + np.outer(dg, dg) / b

    def run(self, frame, step: int):
        pass
