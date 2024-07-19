import numpy as np

from molpy.potential.base import Potential
from molpy.core.space import Free


def F(k, r, r0):
    return -k * (r - r0)


def E(k, r, r0):
    return 0.5 * k * (r - r0) ** 2


class Harmonic(Potential):

    F = F
    E = E

    name = "harmonic"
    type = "bond"
    registered_params = ("k", "r0")
    inputs = ("xyz", "bond_idx", "atomtype")
    outputs = (
        "harmonic_bond_energy",
        "per_atom_harmonic_bond_energy",
        "harmonic_bond_force",
    )

    def __init__(self, k, r0):
        super().__init__()
        self.k = k
        self.r0 = r0

    def forward(self, frame, output: dict):

        box = getattr(frame, "box", Free())
        xyz = frame.atoms.xyz
        bond_idx = frame.topology.bonds
        idx_i = bond_idx[:, 0]
        idx_j = bond_idx[:, 1]
        type_i = frame.atoms.atomtype[idx_i]
        type_j = frame.atoms.atomtype[idx_j]

        dr = box.diff(xyz[idx_j], xyz[idx_i])
        r = np.linalg.norm(dr, axis=-1)

        k_ = self.k["k"][type_i, type_j]
        r0_ = self.r0["r0"][type_i, type_j]
        energy = Harmonic.E(r, k_, r0_)
        output["harmonic_bond_energy"] = energy.sum()
        output["per_atom_harmonic_bond_energy"] = np.zeros(frame.n_atoms)
        output["per_atom_harmonic_bond_energy"][idx_i] = energy / 2
        output["harmonic_bond_force"] = np.zeros((frame.n_atoms, 3))
        output["harmonic_bond_force"][idx_i] = (
            Harmonic.F(r, k_, r0_)[:, np.newaxis] * dr / r[:, np.newaxis]
        )
        return frame, output
