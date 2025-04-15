import numpy as np
import molpack as mpk
import molpy as mp

def write_xyz_traj(filename):
    """Write a trajectory to an xyz file."""
    f = open(filename, "w")

    def writer(points):
        f.write(f"{len(points)}\n")
        f.write("Comment\n")
        for point in points:
            f.write(f"X {point[0]} {point[1]} {point[2]}\n")

    return writer


writer = write_xyz_traj("output.xyz")


from scipy.optimize import minimize


# === Objective wrapper ===
class ObjectiveFunction:
    def __init__(self, constraint):
        self.constraint = constraint

    def value(self, x):
        points = x.reshape(-1, 3)
        writer(points)
        print(np.sum(mp.SphereRegion(sphere_radius, sphere_center).isin(points)))
        penalty = self.constraint.penalty(points)
        return penalty

    def gradient(self, x):
        points = x.reshape(-1, 3)
        grad = self.constraint.dpenalty(points)
        return grad


# === Simulation parameters ===
# x0 = np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]).astype(np.float64).flatten()
N = 100
box_origin = np.array([0, 0, 0])
box_size = np.array([10, 10, 10])
sphere_center = np.array([5, 5, 5])
sphere_radius = 3.0

# === Initialize system ===
np.random.seed(42)
x0 = np.random.uniform(low=0, high=10, size=(N, 3)).astype(np.float64).flatten()

bounds = [(0, 10)] * (3 * N)

# === Define constraints ===
constraints = (
    mpk.MinDistanceConstraint(dmin=2.0)
    & (
        mpk.InsideBoxConstraint(box_size, box_origin)
        & ~mpk.InsideSphereConstraint(sphere_radius, sphere_center)
    )
)


# === Optimize
obj = ObjectiveFunction(constraints)
result = minimize(
    obj.value,
    x0,
    jac=obj.gradient,
    bounds=bounds,
    method="L-BFGS-B",
    tol=1e-6,
    options={
        "gtol": 1e-6,
        "ftol": 1e-6,
    },
)

# === Output xyz file
points = result.x.reshape(-1, 3)
print(np.sum(mp.SphereRegion(sphere_radius, sphere_center).isin(points)))
