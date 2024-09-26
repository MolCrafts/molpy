from copy import deepcopy
from molpy import op
from typing import Callable
import numpy as np
import pyarrow as pa


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

    def to_dict(self):
        d = dict(self)
        return d


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
        d = dict(self)
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
        d = dict(self)
        d["i"] = self.itom['id']
        d["j"] = self.jtom['id']
        d["k"] = self.ktom['id']
        return d


class Dihedral(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def to_dict(self):
        d = dict(self)
        d["i"] = self.itom['id']
        d["j"] = self.jtom['id']
        d["k"] = self.ktom['id']
        d["l"] = self.ltom['id']
        return d


class Improper(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def to_dict(self):
        d = dict(self)
        d["i"] = self.itom['id']
        d["j"] = self.jtom['id']
        d["k"] = self.ktom['id']
        d["l"] = self.ltom['id']
        return d


class MolpyModel(dict):
    pass


class Entities(list):

    def append(self, entity):
        entity['id'] = len(self) + 1
        super().append(entity)

    def keys(self):
        return self[0].keys()
            
    def extend(self, entities):
        for i, en in enumerate(entities, 1):
            en['id'] = len(self) + i
        super().extend(entities)

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
    def from_structs(cls, structs):
        struct = Struct()
        for s in structs:
            struct.union_(s)
        return struct

    def __repr__(self):
        return f"<Struct {len(self['atoms'])} atoms>"

    def add_atom(self, atom: Atom):
        self["atoms"].append(atom)

    def add_bond(self, bond: Bond):
        self["bonds"].append(bond)

    def add_angle(self, angle: Angle):
        self["angles"].append(angle)

    def add_dihedral(self, dihedral: Dihedral):
        self["dihedrals"].append(dihedral)

    def add_improper(self, improper: Improper):
        self["impropers"].append(improper)

    def del_atom(self, atom: Atom):

        self["atoms"].remove(atom)

        for bond in self["bonds"]:
            if atom in {bond.itom, bond.jtom}:
                self["bonds"].remove(bond)

    def del_bond_(self, itom, jtom):
        if isinstance(itom, int) and isinstance(jtom, int):
            itom = self.get_atom_by(lambda atom: atom['id'] == itom)
            jtom = self.get_atom_by(lambda atom: atom['id'] == jtom)
        for bond in self["bonds"]:
            if {bond.itom, bond.jtom} == {itom, jtom}:
                self["bonds"].remove(bond)

    def del_bond(self, itom, jtom):

        new = self.copy()
        new.del_bond_(itom, jtom)
        return new

    def get_atom_by(self, condition: Callable[[Atom], bool]) -> Atom:
        for atom in self["atoms"]:
            if condition(atom):
                return atom

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

    def move(self, r):
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.translate(xyz, r)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self

    def rotate(self, axis, theta):
        for atom in self["atoms"]:
            xyz = np.array([[atom["x"], atom["y"], atom["z"]]])
            xyz = op.rotate(xyz, axis, theta)
            atom["x"], atom["y"], atom["z"] = xyz[0, 0], xyz[0, 1], xyz[0, 2]
        return self
    
    def link(self, other: "Struct", new_bonds: list[Bond]=[]):
        
        new_struct = Struct()
        new_struct.union_(self)
        # TODO: optimize position
        new_struct.union_(other)
        for bond in new_bonds:
            new_struct.add_bond(bond)
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
        frame["atoms"] = pa.Table.from_pylist(atom_dict).sort_by("id")
        if self["bonds"]:
            bond_dict = [bond.to_dict() for bond in self["bonds"]]
            frame["bonds"] = pa.Table.from_pylist(bond_dict).sort_by("id")

        if self["angles"]:
            angle_dict = [angle.to_dict() for angle in self["angles"]]
            frame["angles"] = pa.Table.from_pylist(angle_dict).sort_by("id")

        if self["dihedrals"]:
            dihedral_dict = [dihedral.to_dict() for dihedral in self["dihedrals"]]
            frame["dihedrals"] = pa.Table.from_pylist(dihedral_dict).sort_by("id")

        if self["impropers"]:
            improper_dict = [improper.to_dict() for improper in self["impropers"]]
            frame["impropers"] = pa.Table.from_pylist(improper_dict).sort_by("id")

        frame["props"]["n_atoms"] = len(self["atoms"])
        frame["props"]["n_bonds"] = len(self["bonds"])
        frame["props"]["n_angles"] = len(self["angles"])
        frame["props"]["n_dihedrals"] = len(self["dihedrals"])
        frame["props"]["n_impropers"] = len(self["impropers"])
        frame["props"]["n_atomtypes"] = len(
            np.unique([atom["type_name"] for atom in self["atoms"]])
        )
        frame["props"]["n_bondtypes"] = len(
            np.unique([bond["type_name"] for bond in self["bonds"]])
        )
        frame["props"]["n_angletypes"] = len(
            np.unique([angle["type_name"] for angle in self["angles"]])
        )
        frame["props"]["n_dihedraltypes"] = len(
            np.unique([dihedral["type_name"] for dihedral in self["dihedrals"]])
        )
        frame["props"]["n_impropertypes"] = len(
            np.unique([improper["type_name"] for improper in self["impropers"]])
        )

        return frame

    def get_substruct(self, atom_idx):
        
        substruct = Struct()
        for atom in self["atoms"]:
            if atom['id'] in atom_idx:
                substruct.add_atom(atom)
        
        for bond in self["bonds"]:
            if bond.itom['id'] in atom_idx and bond.jtom['id'] in atom_idx:
                substruct.add_bond(bond)
        
        for angle in self["angles"]:
            if angle.itom['id'] in atom_idx and angle.jtom['id'] in atom_idx and angle.ktom['id'] in atom_idx:
                substruct.add_angle(angle)
        
        for dihedral in self["dihedrals"]:
            if dihedral.itom['id'] in atom_idx and dihedral.jtom['id'] in atom_idx and dihedral.ktom['id'] in atom_idx and dihedral.ltom['id'] in atom_idx:
                substruct.add_dihedral(dihedral)
        
        for improper in self["impropers"]:
            if improper.itom['id'] in atom_idx and improper.jtom['id'] in atom_idx and improper.ktom['id'] in atom_idx and improper.ltom['id'] in atom_idx:
                substruct.add_improper(improper)
        
        return substruct