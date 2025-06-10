from .base import Packer
import nlopt
import numpy as np

class ObjectiveFunction:
    def __init__(self, constraint):
        self.constraint = constraint

    def value(self, x):
        points = x.reshape(-1, 3)
        penalty = self.constraint.penalty(points)
        return penalty

    def gradient(self, x):
        points = x.reshape(-1, 3)
        grad = self.constraint.dpenalty(points)
        return grad

    def __call__(self, x, grad):
        if grad.size > 0:
            grad[:] = self.gradient(x).flatten()

        return self.value(x)


class NloptPacker(Packer):

    def __init__(self, method="LD_MMA"):
        super().__init__()
        self.method = getattr(nlopt, method)
        self.opt = nlopt.opt(self.method, 3 * N)
        self.lb = -np.inf
        self.ub = np.inf
        
    def optimize(self, lb, ub, x0, maxeval=100):
        opt = self.opt
        obj = ObjectiveFunction(constraints)
        opt.set_min_objective(obj)
        opt.set_lower_bounds(np.array([0.0] * (3 * N)))
        opt.set_upper_bounds(np.array([10.0] * (3 * N)))
        opt.set_xtol_rel(1e-4)
        opt.set_maxeval(1000)
        result = opt.optimize(x0)
        isin = mpk.InsideSphereConstraint(
            center=np.array([5, 5, 5]), radius=5.0
        ).region.isin(result.reshape(-1, 3))
        assert isin.sum() == 0, f"{isin.sum()} points are inside the constraint region."