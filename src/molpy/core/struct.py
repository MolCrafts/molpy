from copy import deepcopy
from molpy import op
from typing import Callable
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("molpy-struct")

class Entity(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def clone(self):
        return deepcopy(self)
    
    def to_dict(self):
        return dict(self)

class Atom(Entity):

    def __repr__(self):
        return f"<Atom {self['name']}>"

    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return self['name'] == other['name']

class Bond(Entity):

    def __init__(self, itom: Atom, jtom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom

    def __repr__(self):
        return f"<Bond {self.itom} {self.jtom}>"

    def __eq__(self, other):
        if isinstance(other, Bond):
            return {self.itom, self.jtom} == {other.itom, other.jtom}
        return False

    def __hash__(self):
        return hash((self.itom, self.jtom))
    
    def to_dict(self):
        d = super().to_dict()
        d['i'] = self.itom['id']
        d['j'] = self.jtom['id']
        return d


class Angle(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom

    def to_dict(self):
        return super().to_dict() | dict(i=self.itom['id'], j=self.jtom['id'], k=self.ktom['id'])

class Dihedral(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def to_dict(self):
        return super().to_dict() | dict(i=self.itom['id'], j=self.jtom['id'], k=self.ktom['id'], l=self.ltom['id'])

class Improper(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def to_dict(self):
        return super().to_dict() | dict(i=self.itom['id'], j=self.jtom['id'], k=self.ktom['id'], l=self.ltom['id'])

class MolpyModel(dict):
    pass

class Entities(list):

    def keys(self):
        return self[0].keys()

    def get_by(self, condition: Callable[[Entity], bool]) -> Entity:
        for entity in self:
            if condition(entity):
                return entity

class Struct(MolpyModel):

    def __init__(self):
        super().__init__()
        self["atoms"] = Entities()
        self["bonds"] = Entities()
        self["angles"] = Entities()
        self["dihedrals"] = Entities()
        self["impropers"] = Entities()

    @classmethod
    def from_structs(cls, *structs):
        struct = Struct()
        for s in structs:
            struct.union_(s)
        return struct

    def __repr__(self):
        return f"<Struct {len(self['atoms'])} atoms>"

    def add_atom_(self, atom: Atom):
        self["atoms"].append(atom)
        return self

    def add_bond_(self, bond: Bond):
        self["bonds"].append(bond)
        return self

    def add_angle_(self, angle: Angle):
        self["angles"].append(angle)

    def add_dihedral_(self, dihedral: Dihedral):
        self["dihedrals"].append(dihedral)

    def add_improper_(self, improper: Improper):
        self["impropers"].append(improper)

    def del_atom_(self, atom: Atom):

        self["atoms"].remove(atom)

        for bond in self["bonds"]:
            if atom in {bond.itom, bond.jtom}:
                self["bonds"].remove(bond)

        for angle in self["angles"]:
            if atom in {angle.itom, angle.jtom, angle.ktom}:
                self["angles"].remove(angle)

        for dihedral in self["dihedrals"]:
            if atom in {dihedral.itom, dihedral.jtom, dihedral.ktom, dihedral.ltom}:
                self["dihedrals"].remove(dihedral)

        for improper in self["impropers"]:
            if atom in {improper.itom, improper.jtom, improper.ktom, improper.ltom}:
                self["impropers"].remove(improper)

        return self

    def del_atom(self, atom):
        new = self.copy()
        new.del_atom_(atom)
        return new

    def del_bond_(self, itom, jtom):
        if isinstance(itom, int) and isinstance(jtom, int):
            itom = self.get_atom_by(lambda atom: atom['id'] == itom)
            jtom = self.get_atom_by(lambda atom: atom['id'] == jtom)
        for bond in self["bonds"]:
            if (bond.itom == itom and bond.jtom == jtom) or (bond.itom == jtom and bond.jtom == itom):
                self["bonds"].remove(bond)

    def del_bond(self, itom, jtom):

        new = self.copy()
        new.del_bond_(itom, jtom)
        return new

    def get_atom_by(self, condition: Callable[[Atom], bool]) -> Atom:
        for atom in self["atoms"]:
            if condition(atom):
                return atom
    
    def get_atom_by_id(self, id_):
        return self.get_atom_by(lambda atom: atom['id'] == id_)

    def union(self, other: "Struct") -> "Struct":
        struct = self.copy()
        struct.union_(other)
        return struct

    def union_(self, other: "Struct") -> "Struct":
        self["atoms"].extend(other['atoms'])
        self["bonds"].extend(other['bonds'])
        self["angles"].extend(other['angles'])
        self["dihedrals"].extend(other['dihedrals'])
        self["impropers"].extend(other['impropers'])
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
    
    def link(self, other: "Struct", new_bonds: list[Bond]=[]):
        
        new_struct = Struct.from_structs(self, other)
        
        for bond in new_bonds:
            new_struct.add_bond_(bond)
        return new_struct

    def copy(self):
        return deepcopy(self)

    def split(self, key="molid"):
        unique_id = np.unique(self['atoms'][key])
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

        atom_dict = [atom.to_dict() for atom in self["atoms"]]
        frame["atoms"] = pd.DataFrame(atom_dict).sort_values("id")
        if self["bonds"]:
            bond_dict = [bond.to_dict() for bond in self["bonds"]]
            frame["bonds"] = pd.DataFrame(bond_dict).sort_values("id")

        if self["angles"]:
            angle_dict = [angle.to_dict() for angle in self["angles"]]
            frame["angles"] = pd.DataFrame(angle_dict).sort_values("id")

        if self["dihedrals"]:
            dihedral_dict = [dihedral.to_dict() for dihedral in self["dihedrals"]]
            frame["dihedrals"] = pd.DataFrame(dihedral_dict).sort_values("id")

        if self["impropers"]:
            improper_dict = [improper.to_dict() for improper in self["impropers"]]
            frame["impropers"] = pd.DataFrame(improper_dict).sort_values("id")

        # frame["props"]["n_atoms"] = len(self["atoms"])
        # frame["props"]["n_bonds"] = len(self["bonds"])
        # frame["props"]["n_angles"] = len(self["angles"])
        # frame["props"]["n_dihedrals"] = len(self["dihedrals"])
        # frame["props"]["n_impropers"] = len(self["impropers"])
        # frame["props"]["n_atomtypes"] = len(
        #     np.unique([atom["type"] for atom in self["atoms"]])
        # )
        # frame["props"]["n_bondtypes"] = len(
        #     np.unique([bond["type"] for bond in self["bonds"]])
        # )
        # frame["props"]["n_angletypes"] = len(
        #     np.unique([angle["type"] for angle in self["angles"]])
        # )
        # frame["props"]["n_dihedraltypes"] = len(
        #     np.unique([dihedral["type"] for dihedral in self["dihedrals"]])
        # )
        # frame["props"]["n_impropertypes"] = len(
        #     np.unique([improper["type"] for improper in self["impropers"]])
        # )

        return frame

    def get_substruct(self, atom_idx):
        
        substruct = Struct()
        for atom in self["atoms"]:
            if atom['id'] in atom_idx:
                substruct.add_atom_(atom)
        
        for bond in self["bonds"]:
            if bond.itom['id'] in atom_idx and bond.jtom['id'] in atom_idx:
                substruct.add_bond_(bond)
        
        for angle in self["angles"]:
            if angle.itom['id'] in atom_idx and angle.jtom['id'] in atom_idx and angle.ktom['id'] in atom_idx:
                substruct.add_angle_(angle)
        
        for dihedral in self["dihedrals"]:
            if dihedral.itom['id'] in atom_idx and dihedral.jtom['id'] in atom_idx and dihedral.ktom['id'] in atom_idx and dihedral.ltom['id'] in atom_idx:
                substruct.add_dihedral_(dihedral)
        
        for improper in self["impropers"]:
            if improper.itom['id'] in atom_idx and improper.jtom['id'] in atom_idx and improper.ktom['id'] in atom_idx and improper.ltom['id'] in atom_idx:
                substruct.add_improper_(improper)
        
        return substruct
    
class Structs(list):
    ...

    def to_frame(self):
        from .frame import Frame
        frame = Frame()
        for struct in self:
            frame += struct.to_frame()
        return frame