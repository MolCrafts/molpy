from .space import Box
import molpy as mp
import numpy as np
import pandas as pd
from copy import deepcopy

class Frame(dict):

    def __init__(self, data: dict[str, pd.DataFrame | dict[str, list]] | None = None, **props):
        """ Static data structure for aligning model. The frame is a dictionary-like, multi-DataFrame object, facilitating access data by keys.

        Args:
            data (dict): A dictionary of dataframes.
        """
        if data is not None:
            for key, value in data.items():
                assert isinstance(value, (pd.DataFrame, pd.Series)) or isinstance(value, dict), TypeError("data must be dataframe-like, otherwise assign them by `Frame(name=object)` and call it by `frame.name`")
                self[key] = pd.DataFrame(data=value)

        for k, v in props.items():
            setattr(self, k, v)

        self._box = props["box"] if "box" in props else None

    @classmethod
    def From_frames(cls, others):
        frame = cls()
        for fkey in set([k for other in others for k in other.keys()]):
            frame[fkey] = pd.concat((other[fkey] for other in others), axis=0, ignore_index=True, sort=False)

        # TODO: box same
        frame.box = getattr(others[0], "box")
        return frame


    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box: Box|None):
        assert isinstance(box, Box) or box is None, "box must be an instance of Box or None"
        self._box = box

    @classmethod
    def concat(cls, frames: list["Frame"]) -> "Frame":
        """ Concatenate a list of frames into a single frame. 

            Args:
                frames (list[Frame]): A list of frames.
        """
        frame = cls()
        for key in frames[0].keys():
            if isinstance(frames[0][key], pd.DataFrame):
                frame[key] = pd.concat(
                    [frame[key] for frame in frames if key in frame]
                )

        return frame

    def to_struct(self):
        from .struct import Entities, Struct
        struct = Struct(
            atoms=Entities(),
        )
        atoms = self["atoms"]
        for _, atom in atoms.iterrows():
            struct.add_atom_(mp.Atom(**atom))

        if "bonds" in self:
            struct["bonds"] = Entities()
            bonds = self["bonds"]
            for _, bond in bonds.iterrows():
                i, j = bond.pop("i"), bond.pop("j")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                struct["bonds"].append(
                    mp.Bond(
                        itom,
                        jtom,
                        **{k: v for k, v in bond.items()},
                    )
                )

        if "angles" in self:
            struct["angles"] = Entities()
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
            struct["dihedrals"] = Entities()
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
        elif isinstance(key, tuple) and all(isinstance(k, str) for k in key):
            if len(key) == 2:
                return self[key[0]][key[1]]
            return self[key[0]][list(key[1:])]
        if isinstance(key, (slice, pd.Series, np.ndarray)):
            mask = pd.Series(key, index=self["atoms"].index)
            atoms = self["atoms"][mask]
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
                dict(atoms=atoms,
                bonds=bonds,
                angles=angles,
                dihedrals=dihedrals,
                ),
                box = self.box
            )
        if isinstance(key, tuple):
            return self[key[0]][list(key[1:])]
