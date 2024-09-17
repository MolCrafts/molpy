from copy import deepcopy
from molpy import op
from typing import Callable
import numpy as np
import pyarrow as pa


class Entity(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __eq__(self, other):
        return self.id == other.id

    def clone(self):
        return deepcopy(self)


class Atom(Entity):

    def __init__(self, id: int, *args, **kwargs):
        self.id = id
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"<Atom {self.id}>"

    def __hash__(self):
        return id(self)

    def to_dict(self):
        d = dict(self)
        d["id"] = self.id
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
        d["itom"] = self.itom.id
        d["jtom"] = self.jtom.id
        return d


class Angle(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom


class Dihedral(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom


class Improper(Entity):
    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, ltom: Atom, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom


class MolpyModel(dict):
    pass


class Entities(list):

    def add(self, entity: Entity):
        self.append(entity)

    def keys(self):
        return self[0].keys()

    def get_by_id(self, id: int):
        for entity in self:
            if entity.id == id:
                return entity


class Struct(MolpyModel):

    def __init__(self):
        super().__init__()
        self["atoms"] = Entities()
        self["bonds"] = Entities()
        self["angles"] = Entities()
        self["dihedrals"] = Entities()
        self["impropers"] = Entities()

    def __repr__(self):
        return f"<Struct {len(self['atoms'])} atoms>"

    def add_atom(self, atom: Atom):
        self["atoms"].add(atom)

    def add_bond(self, bond: Bond):
        self["bonds"].add(bond)

    def add_angle(self, angle: Angle):
        self["angles"].add(angle)

    def add_dihedral(self, dihedral: Dihedral):
        self["dihedrals"].add(dihedral)

    def add_improper(self, improper: Improper):
        self["impropers"].add(improper)

    def del_atom(self, atom: Atom):

        self["atoms"].remove(atom)

        for bond in self["bonds"]:
            if atom in {bond.itom, bond.jtom}:
                self["bonds"].remove(bond)

    def get_atom_by_id(self, id: str):
        for atom in self["atoms"]:
            if atom.id == id:
                return atom

    def get_atom(self, condition: Callable[[Atom], bool]) -> Atom:
        for atom in self["atoms"]:
            if condition(atom):
                return atom

    def union(self, other: "Struct") -> "Struct":
        struct = self.copy()
        struct.union_(other)
        return struct

    def union_(self, other: "Struct") -> "Struct":
        self["atoms"].update(other.atoms)
        self["bonds"].update(other.bonds)
        self["angles"].update(other.angles)
        self["dihedrals"].update(other.dihedrals)
        return self

    def move(self, r):
        for atom in self["atoms"]:
            atom["xyz"] = op.translate(atom["xyz"], r)
        return self

    def rotate(self, axis, theta):
        for atom in self["atoms"]:
            atom["xyz"] = op.rotate(atom["xyz"], axis, theta)
        return self

    def split(self, mask, key="molid"):
        unique_id = np.unique(mask)
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
        frame["atoms"] = pa.table(
            {k: [d[k] for d in self["atoms"]] for k in self["atoms"][0].keys()}
        )

        if len(self["bonds"]) != 0:
            bonds = {"i": [], "j": []}
            for bond in self["bonds"]:
                i = bond.itom.id
                j = bond.jtom.id
                bonds["i"].append(i)
                bonds["j"].append(j)
            bonds.update(
                {k: [d[k] for d in self["bonds"]] for k in self["bonds"][0].keys()}
            )
            frame["bonds"] = pa.table(bonds)

        if len(self["angles"]) != 0:
            angles = {"i": [], "j": [], "k": []}
            for angle in self["angles"]:
                i = angle.itom.id
                j = angle.jtom.id
                k = angle.ktom.id
                angles["i"].append(i)
                angles["j"].append(j)
                angles["k"].append(k)

            angles.update(
                {k: [d[k] for d in self["angles"]] for k in self["angles"][0].keys()}
            )
            frame["angles"] = pa.table(angles)

        if len(self["dihedrals"]) != 0:

            dihedrals = {"i": [], "j": [], "k": [], "l": []}
            for dihedral in self["dihedrals"]:
                i = dihedral.itom.id
                j = dihedral.jtom.id
                k = dihedral.ktom.id
                l = dihedral.ltom.id
                dihedrals["i"].append(i)
                dihedrals["j"].append(j)
                dihedrals["k"].append(k)
                dihedrals["l"].append(l)

            dihedrals.update(
                {
                    k: [d[k] for d in self["dihedrals"]]
                    for k in self["dihedrals"][0].keys()
                }
            )
            frame["dihedrals"] = pa.table(dihedrals)

        if len(self["impropers"]) != 0:

            impropers = {"i": [], "j": [], "k": [], "l": []}
            for improper in self["impropers"]:
                i = improper.itom.id
                j = improper.jtom.id
                k = improper.ktom.id
                l = improper.ltom.id
                impropers["i"].append(i)
                impropers["j"].append(j)
                impropers["k"].append(k)
                impropers["l"].append(l)

            impropers.update(
                {
                    k: [d[k] for d in self["impropers"]]
                    for k in self["impropers"][0].keys()
                }
            )
            frame["impropers"] = pa.table(impropers)

        frame['props']['n_atoms'] = len(self['atoms'])
        frame['props']['n_bonds'] = len(self['bonds'])
        frame['props']['n_angles'] = len(self['angles'])
        frame['props']['n_dihedrals'] = len(self['dihedrals'])
        frame['props']['n_impropers'] = len(self['impropers'])
        frame['props']['n_atomtypes'] = len(np.unique([atom['type'] for atom in self['atoms']]))
        frame['props']['n_bondtypes'] = len(np.unique([bond['type'] for bond in self['bonds']]))
        frame['props']['n_angletypes'] = len(np.unique([angle['type'] for angle in self['angles']]))
        frame['props']['n_dihedraltypes'] = len(np.unique([dihedral['type'] for dihedral in self['dihedrals']]))
        frame['props']['n_impropertypes'] = len(np.unique([improper['type'] for improper in self['impropers']]))

        return frame
