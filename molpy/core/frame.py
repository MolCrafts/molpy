# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from typing import List, Optional
from .topology import Topology
from .item import Atom, Bond, Angle, Dihedral
from .box import Box
import numpy as np

class Frame:

    pass

class DynamicFrame(Frame):

    def __init__(self, box:Optional[Box]=None, topo:Optional[Topology]=None, timestep:Optional[int]=None):
        
        self.timestep = timestep
        self._box = box
        self._atoms = {}
        self._bonds = {}
        if topo is None:
            self._topo = Topology()
        else:
            self._topo = topo

    @classmethod
    def from_dict(cls, data:dict[str, np.array], box:Optional[Box]=None, topo:Optional[Topology]=None, timestep:Optional[int]=None):

        dframe = cls(box, topo, timestep)

        for i in zip(*data.values()):
            dframe.add_atom(**{k: v for k, v in zip(data.keys(), i)})

        return dframe

    @property
    def atoms(self):
        return list(self._atoms.values())

    @property
    def bonds(self):
        return list(self._bonds.values())

    @property
    def n_atoms(self):
        return len(self._atoms)

    def add_atom(self, **attribs):

        atom = Atom(**attribs)

        self._atoms[id(atom)] = atom
        self._topo.add_atom(id(atom))

    def del_atom(self, i):
        atom_id = id(self.atoms[i])
        self._atoms.pop(atom_id)
        bond_idx = self._topo.del_atom(atom_id)
        for b in bond_idx:
            self._bonds.pop(b)

    def add_bond(self, i:int, j:int, **attribs)->Bond:

        itom = self.atoms[i]
        jtom = self.atoms[j]

        bond = Bond(itom, jtom, **attribs)
        self._bonds[id(bond)] = bond

        # update topology
        self._topo.add_bond(id(itom), id(jtom), id(bond))

        return bond

    def add_bonds(self, atom_idxs, **attribs):

        n_bonds = len(atom_idxs)
        for i in range(n_bonds):
            self.add_bond(atom_idxs[i][0], atom_idxs[i][1], **{k:v[i] for k, v in attribs.items()})

    def del_bond(self, i, j):
        itom = self.atoms[i]
        jtom = self.atoms[j]        
        bond_id = self._topo.del_bond(id(itom), id(jtom))
        del self._bonds[bond_id]

    def get_bond(self, i, j):
        itom = self.atoms[i]
        jtom = self.atoms[j]    
        bond_id = self._topo.get_bond(id(itom), id(jtom))
        return self._bonds[bond_id]

    @property
    def n_bonds(self):
        return len(self._bonds)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, b):
        self._box = b

    def __getitem__(self, key):

        if isinstance(key, str):
            return [atom[key] for atom in self._atoms]
        elif isinstance(key, (int, slice)):
            return self._atoms[key]

    def to_static(self):

        return StaticFrame.from_atoms(self.atoms, self.box, self._topo)


class StaticFrame(Frame):

    def __init__(self, atoms, box:Optional[Box], topo:Optional[Topology], timestep:Optional[int]=None):
        self._atoms = atoms
        self.timestep = timestep
        self._box = box
        if topo is None:
            self._topo = Topology()
        else:
            self._topo = topo

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, b):
        self._box = b

    def __getitem__(self, key):

        return self._atoms[key]

    @classmethod
    def from_atoms(cls, atoms:List[Atom], box=None, topo=None, timestep=None):

        atom_data = []
        atom_field = atoms[0].keys()
        field_type = {field: np.array(atoms[0][field]).dtype for field in atom_field}
        field_shape = {field: np.array(atoms[0][field]).shape for field in atom_field}

        structured_dtype = np.dtype([(field, field_type[field], field_shape[field]) for field in atom_field])

        for atom in atoms:
            atom_data.append(tuple(atom[field] for field in atom_field))

        return cls(np.array(atom_data, dtype=structured_dtype), box, topo, timestep)
        

    @property
    def n_atoms(self):
        return len(self._atoms)