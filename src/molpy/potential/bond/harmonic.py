
from molpy.potential.base import Potential


class Harmonic(Potential):

    def __init__(self, k, r0):
        super().__init__('harmonic', 'bond')
        self.k = k
        self.r0 = r0

    def forward(self, frame):
        bond_idx = frame.bonds
        bi = bond_idx[:, 0]
        bj = bond_idx[:, 1]
