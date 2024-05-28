import numpy as np

from molpy.potential.base import Potential
from molpy.core.struct import Struct
from molpy.core.space import Box


def F(r, k, r0):
    return -k * (r - r0)


def E(r, k, r0):
    return 0.5 * k * (r - r0) ** 2


class Harmonic(Potential):

    F = F
    E = E

    def __new__(cls, ):
        return super().__new__(cls, "harmonic", "bond", ("k", "r0"), ("xyz", "bond_idx", "atomtype"), ("harmonic_bond_energy", "per_atom_harmonic_bond_energy", "harmonic_bond_force"))

    def __init__(self):
        super().__init__()

    def forward(self, struct: Struct, output: dict, **params):

        box = getattr(struct, "box", Box.free())
        xyz = struct.atoms.xyz
        bond_idx = struct.topology.bonds
        idx_i = bond_idx[:, 0]
        idx_j = bond_idx[:, 1]
        type_i = struct.atoms.atomtype[idx_i]
        type_j = struct.atoms.atomtype[idx_j]

        dr = box.diff(xyz[idx_j], xyz[idx_i])
        r = np.linalg.norm(dr, axis=-1)

        k_ = params["k"][type_i, type_j]
        r0_ = params["r0"][type_i, type_j]
        energy = Harmonic.E(r, k_, r0_)
        output["harmonic_bond_energy"] = energy.sum()
        output["per_atom_harmonic_bond_energy"] = np.zeros(struct.n_atoms)
        output["per_atom_harmonic_bond_energy"][idx_i] = energy / 2
        output["harmonic_bond_force"] = np.zeros((struct.n_atoms, 3))
        output["harmonic_bond_force"][idx_i] = Harmonic.F(r, k_, r0_)[:, np.newaxis] * dr / r[:, np.newaxis]
        return struct, output

    def energy(self, xyz, idx_i, idx_j, type_i, type_j, k, r0):
        return Harmonic.E(xyz, idx_i, idx_j, type_i, type_j, k, r0)

    def force(self, xyz, idx_i, idx_j, type_i, type_j, k, r0):
        return Harmonic.F(xyz, idx_i, idx_j, type_i, type_j, k, r0)
