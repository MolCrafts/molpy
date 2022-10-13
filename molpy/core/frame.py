# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-08-10
# version: 0.0.1

from typing import List, Optional
from .topology import Topology
from .item import Atom, Bond, Angle, Dihedral
from .box import Box
import numpy as np
from numpy.lib import recfunctions as rfn

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

    @classmethod
    def from_sturcture_array(cls, sarray, box=None, timestep=None):

        fields = sarray.dtype.names
        data = {k: sarray[k] for k in fields}
        return cls.from_dict(data, box)

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

        self._atoms[atom.id] = atom
        self._topo.add_atom(atom.id)

    def del_atom(self, i):
        atom_id = self.atoms[i].id
        self._atoms.pop(atom_id)
        bond_idx = self._topo.del_atom(atom_id)
        for b in bond_idx:
            self._bonds.pop(b)

    def add_bond(self, i:int, j:int, **attribs)->Bond:
        """
        add bond by the index of atoms

        Parameters
        ----------
        i : int
            index of atom1
        j : int
            index of atom2

        Returns
        -------
        Bond
            bond object
        """
        itom = self.atoms[i]
        jtom = self.atoms[j]

        bond = Bond(itom, jtom, **attribs)
        self._bonds[bond.id] = bond

        # update topology
        self._topo.add_bond(itom.id, jtom.id, bond.id)

        return bond

    def add_bonds(self, atom_idxs, **attribs):

        n_bonds = len(atom_idxs)
        for i in range(n_bonds):
            self.add_bond(atom_idxs[i][0], atom_idxs[i][1], **{k:v[i] for k, v in attribs.items()})

    def add_bond_by_atom_id(self, id1, id2, **attrib):

        atom1 = self._atoms[id1]
        atom2 = self._atoms[id2]

        bond = Bond(atom1, atom2, **attrib)
        self._bonds[bond.id] = bond

        # update topology
        self._topo.add_bond(atom1.id, atom2.id, bond.id)

        return bond

    def add_bonds_by_atom_id(self, atom_ids, **attribs):

        nbonds = len(atom_ids)
        for i in range(nbonds):
            self.add_bond_by_atom_id(atom_ids[i][0], atom_ids[i][1], **{k:v[i] for k, v in attribs.items()})

    def del_bond(self, i, j):
        itom = self.atoms[i]
        jtom = self.atoms[j]        
        bond_id = self._topo.del_bond(itom.id, jtom.id)
        del self._bonds[bond_id]

    def get_bond(self, i, j):
        itom = self.atoms[i]
        jtom = self.atoms[j]    
        bond_id = self._topo.get_bond(itom.id, jtom.id)
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

    def __init__(self, atom_array, box:Optional[Box], topo:Optional[Topology], timestep:Optional[int]=None):
        self._atoms = atom_array
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

    @classmethod
    def from_dict(cls, atom_dict, box=None, topo=None, timestep=None):

        keys = atom_dict.keys()
        values = atom_dict.values()
        atom_array = np.rec.fromarrays(list(values), names=','.join(keys))

        return cls(atom_array, box, topo, timestep)
        
    @property
    def n_atoms(self):
        return len(self._atoms)

    @property
    def n_bonds(self):
        return self._topo.n_bonds

    def add_bond(self, i, j):
        self._topo.add_bond(i, j, None)

    def append(self, another_frame):
        
        self._atoms = rfn.stack_arrays(
            (self._atoms, another_frame._atoms),
            asrecarray=True
        )

        if another_frame.n_bonds != 0:
            
            for bond in another_frame._topo.bonds+self.n_atoms:
                self._topo.add_bond(bond[0], bond[1], None)
