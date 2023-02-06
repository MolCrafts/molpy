# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from operator import add
from typing import List, Optional
from molpy.core.topology import Topology
import numpy as np
from functools import cached_property, reduce


class Atom(dict):

    def __repr__(self):
        return f"<Atom: {self.name}>"

    def __lt__(self, o):
        return id(self) < id(o)

    def __hash__(self):
        return id(self)

    @property
    def name(self):
        return self.get("name", self.get("type", "unknown"))

    @property
    def xyz(self):
        return self.get("xyz", self.get("positions"))

    @property
    def type(self):
        return self.get("type")

    @property
    def charge(self):
        return self.get("charge", None)

    @property
    def mass(self):
        return self.get("mass", None)


class Bond(dict):
    def __init__(self, i, j, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itom = i
        self.jtom = j

    def __iter__(self):
        return (self.i, self.j)


class Residue:
    def __init__(self, name: str, id: Optional[int] = None):

        self.id = id
        self.name = name
        self._atoms:List[Atom] = []
        self.topology = Topology()

    def __repr__(self):
        return f'<Residue: {self.name}>'

    @cached_property
    def natoms(self):
        return len(self._atoms)

    @cached_property
    def nbonds(self):
        return self.topology.nbonds

    def add_atom(self, atom: Atom):
        self._atoms.append(atom)

    def add_bonds(self, bonds):
        self.topology.add_bonds(bonds)

    @cached_property
    def atoms(self) -> np.ndarray[Atom]:
        return np.array(self._atoms)

    @cached_property
    def connect(self) -> np.ndarray:
        return np.array(self.topology.bonds.get("index", []))

    @cached_property
    def bonds(self) -> np.ndarray[Bond]:
        index = self.connect
        atoms = self.atoms
        bond_atoms = atoms[index]
        bonds = np.zeros((len(bond_atoms)), dtype=object)
        for i, b in enumerate(bond_atoms):
            bonds[i] = Bond(*b)
        return bonds

    def translate(self, vector: np.ndarray):
        for atom in self.atoms:
            atom["xyz"] += vector
        return self

    @cached_property
    def xyz(self):
        return np.array([atom["xyz"] for atom in self.atoms])

class Molecule:
    def __init__(self, name: str, id: Optional[int] = None):

        self.id = id
        self.name = name
        self._atoms: List[Atom] = []
        self.residues: List[Residue] = []
        self.topology = Topology()

    def add_atom(self, atom:Atom):
        self._atoms.append(atom)

    def add_bonds(self, atom:Atom):
        self.topology.add_bonds(atom)

    def add_residue(self, residue:Residue):
        self.residues.append(residue)

    @property
    def natoms(self):
        return reduce(add, [res.natoms for res in self.residues], len(self._atoms))

    @property
    def nbonds(self):
        return reduce(add, [res.nbonds for res in self.residues], self.topology.nbonds)

    @property
    def nresidues(self):
        return len(self.residues)

    @property
    def connect(self) -> np.ndarray:

        connect = np.zeros((self.nbonds, 2), dtype=int)
        cursor = self.topology.nbonds
        if cursor:
            connect[:cursor] = np.array(self.topology.connect)
        for residue in self.residues:
            nbonds = residue.nbonds
            next_cur = cursor + nbonds
            connect[cursor:next_cur] = residue.connect + len(self._atoms)
            cursor = next_cur
        return connect

    @property
    def atoms(self) -> np.ndarray[Atom]:
        atoms = [self._atoms]
        atoms.extend([res.atoms for res in self.residues])
        return np.concatenate(atoms)

    @property
    def bonds(self)->np.ndarray[Bond]:
        index = self.connect
        atoms = self.atoms
        bond_atoms = index[atoms]
        bonds = np.zeros((len(bond_atoms)), dtype=object)
        for i, b in enumerate(bond_atoms):
            bonds[i] = Bond(*b)
        return bonds

    @property
    def xyz(self):
        return np.array([atom["xyz"] for atom in self.atoms])

    def translate(self, dr):
        """
        translate this molcule by displacement vector dr

        Parameters
        ----------
        dr : np.ndarray
            (3, )

        Returns
        -------
        self
        """
        for atom in self._atoms:
            atom["xyz"] += dr
        for residue in self.residues:
            residue.translate(dr)
        return self

    def __repr__(self) -> str:
        return f"<Molecule: {self.name}>"
