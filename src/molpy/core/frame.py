import molpy as mp
from .struct import Struct
import numpy as np
import pandas as pd
from copy import deepcopy

class Frame(dict):

    def __init__(self, *fields):
        for field in fields:
            self[field] = pd.DataFrame()

    @classmethod
    def from_frames(cls, *frames: "Frame") -> "Frame":
        frame = cls()
        for key in frames[0].keys():
            if isinstance(frames[0][key], pd.DataFrame):
                frame[key] = pd.concat(
                    [frame[key] for frame in frames if key in frame]
                )

        # frame["props"]["n_atoms"] = len(frame["atoms"])
        # frame["props"]["n_bonds"] = len(frame["bonds"]) if "bonds" in frame else 0
        # frame["props"]["n_angles"] = len(frame["angles"]) if "angles" in frame else 0
        # frame["props"]["n_dihedrals"] = (
        #     len(frame["dihedrals"]) if "dihedrals" in frame else 0
        # )
        # frame["props"]["n_impropers"] = (
        #     len(frame["impropers"]) if "impropers" in frame else 0
        # )
        # frame["props"]["n_atomtypes"] = len(frame["atoms"]["type"].unique())
        # frame["props"]["n_bondtypes"] = (
        #     len(frame["bonds"]["type"].unique()) if "bonds" in frame else 0
        # )
        # frame["props"]["n_angletypes"] = (
        #     len(frame["angles"]["type"].unique()) if "angles" in frame else 0
        # )
        # frame["props"]["n_dihedraltypes"] = (
        #     len(frame["dihedrals"]["type"].unique()) if "dihedrals" in frame else 0
        # )
        # frame["props"]["n_impropertypes"] = (
        #     len(frame["impropers"]["type"].unique()) if "impropers" in frame else 0
        # )

        return frame
    
    def merge(self, other: 'Frame') -> 'Frame':
        return Frame.from_frames(self, other)

    def to_struct(self):

        struct = Struct()
        atoms = self["atoms"]
        for _, atom in atoms.iterrows():
            struct.add_atom_(mp.Atom(**atom))

        if "bonds" in self:
            bonds = self["bonds"]
            for _, bond in bonds.iterrows():
                i, j = bond.pop("i"), bond.pop("j")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                struct.add_bond_(
                    mp.Bond(
                        itom,
                        jtom,
                        **{k: v for k, v in bond.items()},
                    )
                )

        if "angles" in self:
            angles = self["angles"]
            for _, angle in angles.iterrows():
                i, j, k = angle.pop("i"), angle.pop("j"), angle.pop("k")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                ktom = struct["atoms"].get_by(lambda atom: atom["id"] == k)
                struct.add_angle_(
                    mp.Angle(
                        itom,
                        jtom,
                        ktom,
                        **{
                            k: v
                            for k, v in angle.items()
                        },
                    )
                )

        if "dihedrals" in self:
            dihedrals = self["dihedrals"]
            for _, dihedral in dihedrals.iterrows():
                i, j, k, l = dihedral.pop("i"), dihedral.pop("j"), dihedral.pop("k"), dihedral.pop("l")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                ktom = struct["atoms"].get_by(lambda atom: atom["id"] == k)
                ltom = struct["atoms"].get_by(lambda atom: atom["id"] == l)
                struct.add_dihedral_(
                    mp.Dihedral(
                        itom,
                        jtom,
                        ktom,
                        ltom,
                        **{
                            k: v
                            for k, v in dihedral.items()
                        },
                    )
                )

        return struct

    def split(self, key):

        frames = []
        masks = self["atoms"][key]
        unique_mask = masks.unique()

        for mask in unique_mask:
            frame = Frame()
            atom_mask = masks == mask
            frame["atoms"] = self["atoms"][atom_mask]
            atom_id_of_this_frame = frame["atoms"]["id"]
            if "bonds" in self:
                bond_i = self["bonds"]["i"]
                bond_j = self["bonds"]["j"]
                bond_mask = np.logical_and(
                    np.isin(bond_i, atom_id_of_this_frame),
                    np.isin(bond_j, atom_id_of_this_frame),
                )
                frame["bonds"] = self["bonds"][bond_mask]

            if "angles" in self:
                angle_i = self["angles"]["i"]
                angle_j = self["angles"]["j"]
                angle_k = self["angles"]["k"]
                angle_mask = (
                    np.isin(angle_i, atom_id_of_this_frame)
                    & np.isin(angle_j, atom_id_of_this_frame)
                    & np.isin(angle_k, atom_id_of_this_frame)
                )
                frame["angles"] = self["angles"][angle_mask]

            if "dihedrals" in self:
                dihedral_i = self["dihedrals"]["i"]
                dihedral_j = self["dihedrals"]["j"]
                dihedral_k = self["dihedrals"]["k"]
                dihedral_l = self["dihedrals"]["l"]
                dihedral_mask = (
                    np.isin(dihedral_i, atom_id_of_this_frame)
                    & np.isin(dihedral_j, atom_id_of_this_frame)
                    & np.isin(dihedral_k, atom_id_of_this_frame)
                    & np.isin(dihedral_l, atom_id_of_this_frame)
                )
                frame["dihedrals"] = self["dihedrals"][dihedral_mask]

            if "impropers" in self:
                improper_i = self["impropers"]["i"]
                improper_j = self["impropers"]["j"]
                improper_k = self["impropers"]["k"]
                improper_l = self["impropers"]["l"]
                improper_mask = (
                    np.isin(improper_i, atom_id_of_this_frame)
                    & np.isin(improper_j, atom_id_of_this_frame)
                    & np.isin(improper_k, atom_id_of_this_frame)
                    & np.isin(improper_l, atom_id_of_this_frame)
                )
                frame["impropers"] = self["impropers"][improper_mask]

            frames.append(frame)

        return frames

    def __add__(self, other: 'Frame') -> 'Frame':
        return Frame.from_frames(self, other)
    
    def __mul__(self, n: int) -> list['Frame']:
        return Frame.from_frames(*[self.copy() for _ in range(n)])
    
    def copy(self) -> 'Frame':
        return deepcopy(self)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        if isinstance(key, slice):
            atoms = self["atoms"].iloc[key]
            atom_ids = atoms["id"]
            bond_i = self["bonds"]["i"]
            bond_j = self["bonds"]["j"]
            bond_mask = np.logical_and(
                np.isin(bond_i, atom_ids), np.isin(bond_j, atom_ids)
            )
            bonds = self["bonds"][bond_mask]
            angle_i = self["angles"]["i"]
            angle_j = self["angles"]["j"]
            angle_k = self["angles"]["k"]
            angle_mask = (
                np.isin(angle_i, atom_ids)
                & np.isin(angle_j, atom_ids)
                & np.isin(angle_k, atom_ids)
            )
            angles = self["angles"][angle_mask]
            dihedral_i = self["dihedrals"]["i"]
            dihedral_j = self["dihedrals"]["j"]
            dihedral_k = self["dihedrals"]["k"]
            dihedral_l = self["dihedrals"]["l"]
            dihedral_mask = (
                np.isin(dihedral_i, atom_ids)
                & np.isin(dihedral_j, atom_ids)
                & np.isin(dihedral_k, atom_ids)
                & np.isin(dihedral_l, atom_ids)
            )
            dihedrals = self["dihedrals"][dihedral_mask]
            return Frame(
                atoms=atoms,
                bonds=bonds,
                angles=angles,
                dihedrals=dihedrals,
            )
        if isinstance(key, tuple):
            return self[key[0]][list(key[1:])]
