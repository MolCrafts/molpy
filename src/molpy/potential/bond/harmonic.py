import numpy as np

def F(r, k, r0):
    return -k * (r - r0)

def E(r, k, r0):
    return 0.5 * k * (r - r0) ** 2

class Harmonic:

    F = F
    E = E

    def __init__(self, k:float, r0:float):
        self.k = k
        self.r0 = r0

    def energy(self, pos, idx_i, idx_j):
        r = np.linalg.norm(pos[idx_j] - pos[idx_i])
        return Harmonic.E(r, self.k, self.r0)
    
    def force(self, pos, idx_i, idx_j):
        r = np.linalg.norm(pos[idx_j] - pos[idx_i])
        return Harmonic.F(r, self.k, self.r0)