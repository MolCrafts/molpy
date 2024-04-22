import numpy as np


def F(theta, k, theta0):
    return -k * (theta - theta0)

def E(theta, k, theta0):
    return 0.5 * k * (theta - theta0) ** 2

class Harmonic:

    F = F
    E = E

    def __init__(self, k:float, theta0:float):
        self.k = k
        self.theta0 = theta0

    def energy(self, pos, idx_i, idx_j, idx_k, ):

        rij = pos[idx_j] - pos[idx_i]
        rkj = pos[idx_j] - pos[idx_k]

        rij /= np.linalg.norm(rij)
        rkj /= np.linalg.norm(rkj)

        theta = np.arccos(np.dot(rij, rkj))

        return Harmonic.E(theta, self.k, self.theta0)
    
    def force(self, pos, idx_i, idx_j, idx_k):

        rij = pos[idx_j] - pos[idx_i]
        rkj = pos[idx_j] - pos[idx_k]

        rij /= np.linalg.norm(rij)
        rkj /= np.linalg.norm(rkj)

        theta = np.arccos(np.dot(rij, rkj))

        f = Harmonic.F(theta, self.k, self.theta0)

        return f