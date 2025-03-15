from copy import deepcopy
from itertools import combinations, permutations, product
from molpy.op import rotate_by_rodrigues
from typing import Any, Callable, Literal
import numpy as np
import pandas as pd
import logging
from molpy import op

logger = logging.getLogger("molpy-struct")


def return_copy(func):
    def wrapper(self, *args, **kwargs):
        new = self.copy()
        return func(new, *args, **kwargs)

    return wrapper


class Entity(dict):

    def __call__(self):
        return self.copy()

    def clone(self):
        return deepcopy(self)

    def to_dict(self):
        return dict(self)

    def copy(self):
        return self.clone()
    

class SpatialEntity(Entity):

    @property
    def R(self):
        return self["R"]

    @R.setter
    def R(self, value):
        self["R"] = np.asarray(value)

    def distance_to(self, other):
        return np.linalg.norm(self.R - other.R)

    def translate(self, vector):
        self.R += vector

    def rotate(self, axis, theta):
        self.R = rotate_by_rodrigues(self.R.reshape(1, -1), axis, theta).flatten()


class Atom(SpatialEntity):

    def __repr__(self):
        return f"<Atom {self['name']}>"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self["name"] == other["name"]

    def __lt__(self, other):
        return self["name"] < other["name"]

    @property
    def R(self):
        return np.array([self["x"], self["y"], self["z"]])


class ManyBody(Entity):

    def __init__(self, *_atoms, **kwargs):
        super().__init__(**kwargs)
        self._atoms = _atoms


class Bond(ManyBody):

    def __init__(self, itom: Atom, jtom: Atom, **kwargs):
        itom, jtom = sorted([itom, jtom])
        super().__init__(itom, jtom, **kwargs)

    @property
    def itom(self):
        return self._atoms[0]

    @property
    def jtom(self):
        return self._atoms[1]

    def __repr__(self):
        return f"<Bond {self.itom} {self.jtom}>"

    def __eq__(self, other):
        if isinstance(other, Bond):
            return (self.itom == other.itom and self.jtom == other.jtom) or (
                self.itom == other.jtom and self.jtom == other.itom
            )
        return False

    def __hash__(self):
        return hash((self.itom, self.jtom))

    def to_dict(self):
        d = super().to_dict()
        d["i"] = self.itom["id"]
        d["j"] = self.jtom["id"]
        return d


class Angle(ManyBody):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, **kwargs):
        itom, ktom = sorted([itom, ktom])
        super().__init__(itom, jtom, ktom, **kwargs)

    @property
    def itom(self):
        return self._atoms[0]

    @property
    def jtom(self):
        return self._atoms[1]

    @property
    def ktom(self):
        return self._atoms[2]

    def to_dict(self):
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"]
        )


class Dihedral(ManyBody):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        if jtom > ktom:
            jtom, ktom = ktom, jtom
            itom, ltom = ltom, itom
        super().__init__(itom, jtom, ktom, ltom, **kwargs)

    @property
    def itom(self):
        return self._atoms[0]

    @property
    def jtom(self):
        return self._atoms[1]

    @property
    def ktom(self):
        return self._atoms[2]

    @property
    def ltom(self):
        return self._atoms[3]

    def to_dict(self):
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"], l=self.ltom["id"]
        )


class Improper(ManyBody):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, **kwargs):
        jtom, ktom, ltom = sorted([jtom, ktom, ltom])
        super().__init__(itom, jtom, ktom, ltom, **kwargs)

    @property
    def itom(self):
        return self._atoms[0]

    @property
    def jtom(self):
        return self._atoms[1]

    @property
    def ktom(self):
        return self._atoms[2]

    @property
    def ltom(self):
        return self._atoms[3]

    def to_dict(self):
        return super().to_dict() | dict(
            i=self.itom["id"], j=self.jtom["id"], k=self.ktom["id"], l=self.ltom["id"]
        )


class Entities(list):

    def keys(self):
        return self[0].keys()

    def get_by(self, condition: Callable[[Entity], bool]) -> Entity:
        for entity in self:
            if condition(entity):
                return entity

    def set(self, key, value):
        for entity in self:
            entity[key] = value
        return self

    def offset(self, key, value):
        for entity in self:
            entity[key] += value


class AtomEntities(Entities): ...


class Struct(SpatialEntity):

    def __init__(self, **entities):
        super().__init__(**entities)

    @classmethod
    def from_structs(cls, *structs):
        struct = Struct()
        for s in structs:
            struct.union_(s) 
        return struct

    def __repr__(self):
        return f"<Struct {len(self['atoms'])} atoms>"

    def __deepcopy__(self, memo):

        atom_map = {id(atom): atom.copy() for atom in self["atoms"]}
        new = Struct()
        for key, value in self.items():
            new[key] = Entities()
        new["atoms"] = Entities(atom_map.values())
        for key, value in self.items():
            if key == "atoms":
                continue
            for v in value:
                if isinstance(v, ManyBody):
                    try:
                        new[key].append(
                            v.__class__(*[atom_map[id(atom)] for atom in v._atoms], **v)
                        )
                    except KeyError:
                        raise KeyError(f"Atom not found in atom_map: {v._atoms}")
        return new

    def add_atom_(self, atom: Atom):
        self["atoms"].append(atom)
        return self

    def add_bond_(self, iname, jname, **kwargs):
        itom = self.get_atom_by(lambda atom: atom["name"] == iname)
        jtom = self.get_atom_by(lambda atom: atom["name"] == jname)
        bond = Bond(itom, jtom, **kwargs)
        self["bonds"].append(bond)
        return self

    @return_copy
    def add_atom(copy, atom: Atom):
        return copy.add_atom_(atom)

    @return_copy
    def add_bond(copy, iname, jname, **kwargs):
        return copy.add_bond_(iname, jname, **kwargs)

    def add_angle_(self, angle: Angle):
        self["angles"].append(angle)

    def add_dihedral_(self, dihedral: Dihedral):
        self["dihedrals"].append(dihedral)

    def add_improper_(self, improper: Improper):
        self["impropers"].append(improper)

    def del_atom_(self, atom: Atom):

        self["atoms"].remove(atom)

        self.unlink_(atom)

        return self

    @return_copy
    def del_atom(copy, atom):
        return copy.del_atom_(atom)

    def unlink_(self, atom):

        atom = self.get_atom_by_name(atom)

        if "bonds" in self:
            bo = []
            for bond in self["bonds"]:
                if atom not in {bond.itom, bond.jtom}:
                    bo.append(bond)
            self["bonds"] = bo

        if "angles" in self:
            ang = []
            for angle in self["angles"]:
                if atom not in {angle.itom, angle.jtom, angle.ktom}:
                    ang.append(angle)
            self["angles"] = ang

        if "dihedrals" in self:
            dihe = []
            for dihedral in self["dihedrals"]:
                if atom not in {
                    dihedral.itom,
                    dihedral.jtom,
                    dihedral.ktom,
                    dihedral.ltom,
                }:
                    dihe.append(dihedral)
            self["dihedrals"] = dihe

        if "impropers" in self:
            imp = []
            for improper in self["impropers"]:
                if atom not in {
                    improper.itom,
                    improper.jtom,
                    improper.ktom,
                    improper.ltom,
                }:
                    imp.append(improper)
            self["impropers"] = imp

        return self

    @return_copy
    def unlink(copy, atom):
        return copy.unlink_(atom)

    @return_copy
    def add_atom(copy, atom: Atom):
        return copy.add_atom_(atom)

    def del_bond_(self, itom, jtom):
        if isinstance(itom, str) and isinstance(jtom, str):
            itom = self.get_atom_by(lambda atom: atom["name"] == itom)
            jtom = self.get_atom_by(lambda atom: atom["name"] == jtom)
        for bond in self["bonds"]:
            if (bond.itom == itom and bond.jtom == jtom) or (
                bond.itom == jtom and bond.jtom == itom
            ):
                self["bonds"].remove(bond)
        return self

    @return_copy
    def del_bond(copy, itom, jtom):
        return copy.del_bond_(itom, jtom)

    def get_atom_by(self, condition: Callable[[Atom], bool]) -> Atom:
        for atom in self["atoms"]:
            if condition(atom):
                return atom

    def get_atom_by_id(self, id_):
        return self.get_atom_by(lambda atom: atom["id"] == id_)

    def get_atom_by_name(self, name):
        return self.get_atom_by(lambda atom: atom["name"] == name)

    @return_copy
    def union(copy, other: "Struct") -> "Struct":
        return copy.union_(other)

    def union_(self, other: "Struct") -> "Struct":
        for key, value in other.items():
            if key not in self:
                self[key] = Entities()
            self[key] += value

        return self

    def __add__(self, other: "Struct") -> "Struct":

        return self.union(other)

    def __mul__(self, n: int) -> list["Struct"]:
        return [self.copy() for _ in range(n)]

    def move_(self, r):
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.translate(xyz, r)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self

    def rotate_(self, axis, theta):
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.rotate(xyz, axis, theta)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self

    @return_copy
    def link(copy, from_, to_):
        return copy.link_(from_, to_)

    def get_bonds_by_atom(self, atom):
        return Entities(
            [bond for bond in self["bonds"] if atom in [bond.itom, bond.jtom]]
        )

    def copy(self):
        return deepcopy(self)

    def split(self, key="molid"):
        unique_id = np.unique(self["atoms"][key])
        structs = []

        for id_ in unique_id:
            sub_struct = Struct()
            for atom in self["atoms"]:
                if atom[key] == id_:
                    sub_struct.add_atom(atom)

            for bond in self["bonds"]:
                if {bond.itom[key], bond.jtom[key]} == {id_}:
                    sub_struct.add_bond(bond)

            for angle in self["angles"]:
                if {angle.itom[key], angle.jtom[key], angle.ktom[key]} == {id_}:
                    sub_struct.add_angle(angle)

            for dihedral in self["dihedrals"]:
                if {
                    dihedral.itom[key],
                    dihedral.jtom[key],
                    dihedral.ktom[key],
                    dihedral.ltom[key],
                } == {id_}:
                    sub_struct.add_dihedral(dihedral)

            for improper in self["impropers"]:
                if {
                    improper.itom[key],
                    improper.jtom[key],
                    improper.ktom[key],
                    improper.ltom[key],
                } == {id_}:
                    sub_struct.add_improper(improper)

            structs.append(sub_struct)
        return structs

    def to_frame(self):

        from .frame import Frame

        frame = Frame()

        struct = self.copy()
        for i, atom in enumerate(struct["atoms"]):
            atom["id"] = i

        frame["atoms"] = pd.DataFrame([atom.to_dict() for atom in struct["atoms"]])
        if "bonds" in struct and len(struct["bonds"]) > 0:
            bond_dict = [bond.to_dict() for bond in struct["bonds"]]
            frame["bonds"] = pd.DataFrame(bond_dict)
            frame["bonds"]["id"] = range(len(frame["bonds"]))

        if "angles" in struct and len(struct["angles"]) > 0:
            angle_dict = [angle.to_dict() for angle in struct["angles"]]
            frame["angles"] = pd.DataFrame(angle_dict)
            frame["angles"]["id"] = range(len(frame["angles"]))

        if "dihedrals" in struct and len(struct["dihedrals"]) > 0:
            dihedral_dict = [dihedral.to_dict() for dihedral in struct["dihedrals"]]
            frame["dihedrals"] = pd.DataFrame(dihedral_dict)
            frame["dihedrals"]["id"] = range(len(frame["dihedrals"]))

        if "impropers" in struct and len(struct["impropers"]) > 0:
            improper_dict = [improper.to_dict() for improper in struct["impropers"]]
            frame["impropers"] = pd.DataFrame(improper_dict)

        return frame

    def get_substruct(self, atom_idx):
        atom_idx = sorted(atom_idx)
        substruct = Struct()
        for atom in self["atoms"]:
            if atom["id"] in atom_idx:
                substruct.add_atom_(atom)

        for bond in self["bonds"]:
            if bond.itom["id"] in atom_idx and bond.jtom["id"] in atom_idx:
                substruct.add_bond_(bond)

        for angle in self["angles"]:
            if (
                angle.itom["id"] in atom_idx
                and angle.jtom["id"] in atom_idx
                and angle.ktom["id"] in atom_idx
            ):
                substruct.add_angle_(angle)

        for dihedral in self["dihedrals"]:
            if (
                dihedral.itom["id"] in atom_idx
                and dihedral.jtom["id"] in atom_idx
                and dihedral.ktom["id"] in atom_idx
                and dihedral.ltom["id"] in atom_idx
            ):
                substruct.add_dihedral_(dihedral)

        return substruct

    def get_topology(self):
        from .topology import Topology

        topo = Topology()
        atoms = {atom: i for i, atom in enumerate(self["atoms"])}
        atom_attrs = {}
        if all("number" in atom for atom in self["atoms"]):
            atom_attrs["number"] = [atom["number"] for atom in self["atoms"]]
        if all("name" in atom for atom in self["atoms"]):
            atom_attrs["name"] = [atom["name"] for atom in self["atoms"]]

        # TODO: atom name if no number
        topo.add_atoms(len(atoms), **atom_attrs)
        bonds = self["bonds"]
        topo.add_bonds([(atoms[bond.itom], atoms[bond.jtom]) for bond in bonds])
        return topo

    def get_segment_(self, mask: list, key: Literal["name", "id"] = "name", name: str =""):
        atoms = Entities([atom for atom in self["atoms"] if atom[key] in mask])
        bonds = Entities(
            [
                bond
                for bond in self["bonds"]
                if bond.itom[key] in mask and bond.jtom[key] in mask
            ]
        )
        angles = Entities(
            [
                angle
                for angle in self["angles"]
                if angle.itom[key] in mask
                and angle.jtom[key] in mask
                and angle.ktom[key] in mask
            ]
        )
        dihedrals = Entities(
            [
                dihedral
                for dihedral in self["dihedrals"]
                if dihedral.itom[key] in mask
                and dihedral.jtom[key] in mask
                and dihedral.ktom[key] in mask
                and dihedral.ltom[key] in mask
            ]
        )
        return Segment(name=name, atoms=atoms, bonds=bonds, angles=angles, dihedrals=dihedrals)
    
    @return_copy
    def get_segment(copy, mask: list, key: Literal["name", "id"] = "name", name:str=""):
        return copy.get_segment_(mask, key, name=name)


class StructProxy(Struct): ...

class Segment(StructProxy): ...
