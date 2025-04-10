import numpy as np
from tqdm import tqdm
import molpy as mp

class Fix:
    ...

class Dump(Fix):

    def __call__(self, frame, status):
        ...

class DumpXYZ(Dump):

    def __init__(self, fpath: str):
        self.fpath = fpath
        self.writer = mp.io.trajectory.XYZTrajectoryWriter(fpath)

    def __call__(self, frame, status):
        frame['step'] = status['nstep']
        self.writer.write_frame(mp.System(frame=frame, box=mp.Box()))

    def __del__(self):
        self.writer.close()

class BaseOptimizer:

    def __init__(self, potential):

        self.potential = potential

        self.status = {
            "nstep": 0,  # current step
        }
        self.fixes = {}

    def fix(self, event, frame, status):
        for fix in self.fixes.get(event, []):
            fix(frame, status)

    def run(self, frame, step: int, fix: np.ndarray = None):

        for _ in tqdm(self.irun(frame, step, fix)):
            ...

    def irun(self, input_, step, fix):


        frame = input_.copy()

        if fix is None:
            fix = np.zeros(frame["atoms"].shape[0], dtype=bool)
        else:
            fix = np.array(fix, dtype=bool)
        self.initialize(frame)

        while self.status["nstep"] < step:

            self.fix("before_step", frame, self.status)
            new_status = self.step(frame, fix)
            self.fix("after_step", frame, self.status)

            self.status["nstep"] += 1

            if self.early_stop(new_status, self.status):
                break

            self.status.update(new_status)

            yield new_status

    def early_stop(self, new_status, status):
        pass

    def initialize(self):
        pass

    def step(self):
        pass

    def register_fix(self, event, fix):
        if event not in self.fixes:
            self.fixes[event] = []
        self.fixes[event].append(fix)


class BFGS(BaseOptimizer):

    def __init__(self, potential, alpha: float = 70.0, step_limit: float = 1.0):
        super().__init__(potential=potential)
        self.alpha = alpha
        self.maxstep = step_limit

    def initialize(self, frame):

        n_atoms = len(frame["atoms"])

        # initial hessian
        self.H0 = np.eye(3 * n_atoms) * self.alpha

        self.H = None
        self.pos0 = None
        self.forces0 = None

    def step(self, frame, fix) -> dict:

        xyz = frame["atoms"][["x", "y", "z"]].to_numpy()
        forces = self.potential.calc_force(frame)
        forces[fix] = 0

        dpos, steplengths = self.prepare_step(xyz, forces)
        dpos = self.determine_step(dpos, steplengths)
        frame["atoms"][["x", "y", "z"]] = xyz + dpos

        return {"forces": forces}

    def early_stop(self, new_status, last_status):

        if (
            "forces" in last_status
            and np.all(np.linalg.norm(new_status["forces"] - last_status["forces"], axis=-1) < 1e-6)
        ):
            return True
        return False

    def prepare_step(self, pos, forces):

        forces = forces.reshape(-1)
        self.update(pos.flat, forces, self.pos0, self.forces0)
        omega, V = np.linalg.eigh(self.H)

        dpos = np.dot(V, np.dot(forces, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dpos**2).sum(1) ** 0.5
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
