from collections.abc import MutableMapping
from copy import deepcopy
from typing import Any, Literal, Sequence, TYPE_CHECKING

import numpy as np
from nesteddict import ArrayDict, NestDict

import molpy as mp
from .box import Box
from molpy.core.utils import TagApplyer

if TYPE_CHECKING:
    from .struct import Struct


class Frame(NestDict):

    box: Box | None = None

    def __new__(cls, data: dict[str, Any] = {}, *, style="atomic") -> "Frame":

        if cls is Frame and style == "atomic":
            return AllAtomFrame.__new__(AllAtomFrame, data) 
        return super().__new__(cls)

    def __init__(self, data: dict[str, Any] = {}, *args, **kwargs):
        """Static data structure for aligning model. The frame is a dictionary-like, multi-DataFrame object, facilitating access data by keys.

        Args:
            data (dict): A dictionary of dataframes.
        """
        super().__init__(data)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.keys()}>"

    @classmethod
    def from_frames(cls, others: Sequence["Frame"]) -> "Frame":
        f = others[0].copy()
        for other in others[1:]:
            f = f.concat(other)
        return f

    @classmethod
    def from_structs(cls, structs):
        frame = cls()

        atom_dicts = []
        bond_dicts = []
        bond_index = []
        tager = TagApplyer()
        for struct in structs:
            if "bonds" in struct:
                topo = struct.get_topology()
                bond_dicts.extend(
                    [bond.to_dict() for bond in struct.bonds]
                )
                bond_index.append(
                    topo.bonds + len(atom_dicts)
                )

            atom_dicts.extend(
                [atom.to_dict() for atom in struct.atoms]
            )
            tager.update_dollar_counter()
            tager.apply_tags(atom_dicts)
            tager.apply_tags(bond_dicts)

        frame["atoms"] = ArrayDict.from_dicts(atom_dicts)
        frame["bonds"] = ArrayDict.from_dicts(bond_dicts)
        bond_index = np.concatenate(bond_index)
        frame["bonds"]["i"] = bond_index[:, 0]
        frame["bonds"]["j"] = bond_index[:, 1]
        return frame

    def __len__(self):
        """Return the number of atoms in the frame."""
        return len(self["atoms"])

    def __add__(self, other: "Frame") -> "Frame":
        return self.concat(other)

    def __mul__(self, n: int) -> list["Frame"]:
        return Frame.from_frames([self.copy() for _ in range(n)])

    def copy(self) -> "Frame":
        return deepcopy(self)


class AllAtomMixin(MutableMapping[Literal["atoms", "bonds", "angles", "dihedrals", "impropers"], ArrayDict]):

    def split(self, masks: list[bool] | list[int] | np.ndarray) -> list["Frame"]:

        frames = []
        masks = np.array(masks)
        if masks.dtype == bool:
            unique_mask = [masks]
        else:
            unique_mask = [masks == i for i in np.unique(masks)]

        for mask in unique_mask:
            frame = self.__class__()
            frame["atoms"] = self["atoms"][mask]
            atom_id_of_this_frame = frame["atoms"]["id"]
            if self["bonds"]:
                bond_i = self["bonds"]["i"]
                bond_j = self["bonds"]["j"]
                bond_mask = np.logical_and(
                    np.isin(bond_i, atom_id_of_this_frame),
                    np.isin(bond_j, atom_id_of_this_frame),
                )
                frame["bonds"] = self["bonds"][bond_mask]

            if self["angles"]:
                angle_i = self["angles"]["i"]
                angle_j = self["angles"]["j"]
                angle_k = self["angles"]["k"]
                angle_mask = (
                    np.isin(angle_i, atom_id_of_this_frame)
                    & np.isin(angle_j, atom_id_of_this_frame)
                    & np.isin(angle_k, atom_id_of_this_frame)
                )
                frame["angles"] = self["angles"][angle_mask]

            if self["dihedrals"]:
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

            if self["impropers"]:
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

    def to_struct(self):
        from .struct import Entities, Struct

        struct = Struct()
        atoms = self["atoms"]
        for atom in atoms.iterrows():
            struct.def_atom(**atom)

        if "bonds" in self:
            struct["bonds"] = Entities()
            bonds = self["bonds"]
            for bond in bonds.iterrows():
                i, j = bond.pop("i"), bond.pop("j")
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                struct["bonds"].add(
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
                struct["angles"].add(
                    mp.Angle(
                        itom,
                        jtom,
                        ktom,
                        **{k: v for k, v in angle.items()},
                    )
                )

        if "dihedrals" in self:
            struct["dihedrals"] = Entities()
            dihedrals = self["dihedrals"]
            for _, dihedral in dihedrals.iterrows():
                i, j, k, l = (
                    dihedral.pop("i"),
                    dihedral.pop("j"),
                    dihedral.pop("k"),
                    dihedral.pop("l"),
                )
                itom = struct["atoms"].get_by(lambda atom: atom["id"] == i)
                jtom = struct["atoms"].get_by(lambda atom: atom["id"] == j)
                ktom = struct["atoms"].get_by(lambda atom: atom["id"] == k)
                ltom = struct["atoms"].get_by(lambda atom: atom["id"] == l)
                struct["dihedrals"].add(
                    mp.Dihedral(
                        itom,
                        jtom,
                        ktom,
                        ltom,
                        **{k: v for k, v in dihedral.items()},
                    )
                )

        return struct


class AllAtomFrame(Frame, AllAtomMixin):
    """A frame that contains atomistic infomation. It is a subclass of Frame and implements the AllAtomMixin interface."""
    def __init__(self, data: dict[str, Any] = {}, *args, **kwargs):
        """Initialize the AllAtomFrame with data.

        Args:
            data (dict): A dictionary of dataframes.
        """
        for key in ["atoms", "bonds", "angles", "dihedrals", "impropers"]:
            if key not in data:
                data[key] = ArrayDict()
        super().__init__(data)