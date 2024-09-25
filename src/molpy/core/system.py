import numpy as np
from .space import Box
from .frame import Frame
from .forcefield import ForceField
import pyarrow as pa


class System:

    def __init__(
        self,
        box: Box  | None = None,
        ff: ForceField | None = None,
        frame: Frame | None = None,
    ):
        self.box = box or Box()
        self.forcefield = ff or ForceField()
        self.frame = frame or Frame()

    def simplify_types(self):

        # simplify bond type
        per_bond_type = self.frame["bonds"]["type"].to_numpy()
        bond_i = self.frame["bonds"]["i"].to_numpy()
        bond_j = self.frame["bonds"]["j"].to_numpy()

        bond_i_type = self.frame["atoms"]["type"].to_numpy()[
            bond_i - 1
        ]  # assume atoms id is ordered and no missing, or need to get atom by id
        bond_j_type = self.frame["atoms"]["type"].to_numpy()[bond_j - 1]

        bond_atom_types = np.stack([bond_i_type, bond_j_type], axis=1)
        bond_atom_types = np.sort(bond_atom_types, axis=1)

        unique_bond_atom_types, unique_bond_idx = np.unique(
            bond_atom_types, axis=0, return_index=True
        )

        matches = (bond_atom_types[:, None, 0] == unique_bond_atom_types[:, 0]) & (
            bond_atom_types[:, None, 1] == unique_bond_atom_types[:, 1]
        )  # shape: (n_inputs, which_unique), kind of one hot
        new_mapping = unique_bond_idx[matches.argmax(axis=1)]

        self.frame["bonds"] = self.frame["bonds"].set_column(self.frame["bonds"].schema.get_field_index("type"), "type", [per_bond_type[new_mapping]])

        bond_types = self.forcefield.bondstyles[0].types
        new_bond_types = []
        for bondtype in bond_types:
            if bondtype.name in new_mapping:
                new_bond_types.append(bondtype)
        self.forcefield.bondstyles[0].types = new_bond_types

        # simplify angle type
        per_angle_type = self.frame["angles"]["type"].to_numpy()
        angle_i = self.frame["angles"]["i"].to_numpy()
        angle_j = self.frame["angles"]["j"].to_numpy()
        angle_k = self.frame["angles"]["k"].to_numpy()

        angle_i_type = self.frame["atoms"]["type"].to_numpy()[angle_i - 1]
        angle_j_type = self.frame["atoms"]["type"].to_numpy()[angle_j - 1]
        angle_k_type = self.frame["atoms"]["type"].to_numpy()[angle_k - 1]

        angle_atom_types = np.stack([angle_i_type, angle_k_type], axis=1)
        angle_atom_types = np.sort(angle_atom_types, axis=1)
        angle_atom_types = np.stack(
            [angle_atom_types[:, 0], angle_j_type, angle_atom_types[:, 1]], axis=1
        )

        unique_angle_atom_types, unique_angle_idx = np.unique(
            angle_atom_types, axis=0, return_index=True
        )
        matches = (angle_atom_types[:, None, 1] == unique_angle_atom_types[:, 1]) & (
            (
                (angle_atom_types[:, None, 0] == unique_angle_atom_types[:, 0])
                & (angle_atom_types[:, None, 2] == unique_angle_atom_types[:, 2])
            )
            | (
                (angle_atom_types[:, None, 0] == unique_angle_atom_types[:, 2])
                & (angle_atom_types[:, None, 2] == unique_angle_atom_types[:, 0])
            )
        )

        new_mapping = unique_angle_idx[matches.argmax(axis=1)]
        self.frame["angles"] = self.frame["angles"].set_column(self.frame["angles"].schema.get_field_index("type"), "type", [per_angle_type[new_mapping]])

        angle_types = self.forcefield.anglestyles[0].types
        new_angle_types = []
        for angletype in angle_types:
            if angletype.name in new_mapping:
                new_angle_types.append(angletype)
        self.forcefield.anglestyles[0].types = new_angle_types

        # simplify dihedral type
        per_dihedral_type = self.frame["dihedrals"]["type"].to_numpy()
        dihedral_i = self.frame["dihedrals"]["i"].to_numpy()
        dihedral_j = self.frame["dihedrals"]["j"].to_numpy()
        dihedral_k = self.frame["dihedrals"]["k"].to_numpy()
        dihedral_l = self.frame["dihedrals"]["l"].to_numpy()

        dihedral_i_type = self.frame["atoms"]["type"].to_numpy()[dihedral_i - 1]
        dihedral_j_type = self.frame["atoms"]["type"].to_numpy()[dihedral_j - 1]
        dihedral_k_type = self.frame["atoms"]["type"].to_numpy()[dihedral_k - 1]
        dihedral_l_type = self.frame["atoms"]["type"].to_numpy()[dihedral_l - 1]

        dihedral_atom_types = np.stack(
            [dihedral_i_type, dihedral_j_type, dihedral_k_type, dihedral_l_type], axis=1
        )
        unique_dihe_atom_types, unique_dihe_idx = np.unique(
            dihedral_atom_types, axis=0, return_index=True
        )
        matches = (
            np.all(dihedral_atom_types[:, None, :] == unique_dihe_atom_types, axis=-1) & 
            np.all(dihedral_atom_types[:, None, ::-1] == unique_dihe_atom_types, axis=-1)
        )
        new_mapping = unique_dihe_idx[matches.argmax(axis=1)]
        self.frame["dihedrals"] = self.frame["dihedrals"].set_column(self.frame["dihedrals"].schema.get_field_index("type"), "type", [per_dihedral_type[new_mapping]])

        dihedral_types = self.forcefield.dihedralstyles[0].types
        new_dihedral_types = []
        for dihedraltype in dihedral_types:
            if dihedraltype.name in new_mapping:
                new_dihedral_types.append(dihedraltype)
        self.forcefield.dihedralstyles[0].types = new_dihedral_types
