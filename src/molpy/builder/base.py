import numpy as np
from molpy import Alias
from typing import Any, Collection
import molpy as mp
from molpy.core.struct import Structure, Struct
from molpy.core.topology import Topology


class Item(dict[str, Any]):

    def __getattr__(self, alias: str) -> Any:
        return self[Alias.get(alias).key]

    def __setattr__(self, alias: str, value: Any) -> None:
        self[Alias.get(alias).key] = value

    # def __getitem__(self, key: str) -> Any:
    #     return super().__getitem__(key)

    # def __setitem__(self, key: str, value: Any) -> None:
    #     return super().__setitem__(key, value)


class ItemList(list[Item]):

    def __getattr__(self, alias: str) -> np.ndarray:
        return np.array([getattr(item, alias) for item in self])
    
    def __setattr__(self, alias: str, value: np.ndarray) -> None:
        for item, v in zip(self, value):
            setattr(item, alias, v)

    def __getitem__(self, key: str|slice) -> np.ndarray:
        if isinstance(key, str):
            return np.array([item[key] for item in self])
        else:
            return super().__getitem__(key)
        
    def __setitem__(self, key: str|slice, value: np.ndarray|Item) -> None:
        if isinstance(key, str):
            for item, v in zip(self, value):
                item[key] = v
        else:
            return super().__setitem__(key, value)
        
class Atom(Item):

    def __repr__(self):

        return f"<Atom: {super().__repr__()}>"


class Bond(Item):

    def __init__(self, itom: Atom, jtom: Atom, **props):

        super().__init__(**props)

        self.itom = itom
        self.jtom = jtom

    def __repr__(self):

        return f"<Bond: {super().__repr__()}>"
    
class Angle(Item):

    def __init__(self, itom: Atom, jtom: Atom, ktom: Atom, **props):

        super().__init__(**props)

        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom

    def __repr__(self):

        return f"<Angle: {super().__repr__()}>"


class DynamicStruct(Structure):

    def __init__(self, n_atoms: int = 0, name:str="", *args, **kwargs):

        super().__init__(name)

        self._atoms: ItemList[Atom] = ItemList()
        self._bonds = ItemList()
        self._angles = ItemList()
        self._dihedrals = ItemList()

        self.name = name

        self._topology = Topology(
            n_atoms,
        )

        self._struct_mask = []
        self._n_struct = 0

    @classmethod
    def join(cls, structs: Collection["DynamicStruct"]) -> "DynamicStruct":
        # Type consistency check
        assert all(isinstance(struct, cls) for struct in structs), TypeError(
            "All structs must be of the same type"
        )
        # Create a new struct
        struct = cls()
        for s in structs:
            struct.union(s)
        return struct

    @property
    def topology(self):
        return self._topology

    @property
    def n_atoms(self):
        return len(self._atoms)

    @property
    def atoms(self) -> ItemList[Atom]:
        return self._atoms

    @property
    def bonds(self):
        return self._bonds

    @property
    def angles(self):
        return self._angles

    @property
    def dihedrals(self):
        return self._dihedrals

    @topology.setter
    def topology(self, topology):
        self._topology = topology

    def add_atom(self, atom=None, **props):
        if atom:
            self._atoms.append(atom)
        else:
            self._atoms.append(Atom(**props))
        self._struct_mask.append(self._n_struct)

    def add_bond(self, idx_i:int, idx_j:int, **props):
        self._bonds.append(Bond(self.atoms[idx_i], self.atoms[idx_j], **props))

    def add_angle(self, idx_i:int, idx_j:int, idx_k:int, **props):
        self._angles.append(Angle(self.atoms[idx_i], self.atoms[idx_j], self.atoms[idx_k], **props))

    def add_struct(self, struct: "DynamicStruct"):

        self.union(struct)
        self._n_struct += 1
        self._struct_mask.extend([self._n_struct] * struct.n_atoms)

    def union(self, other: "DynamicStruct") -> "DynamicStruct":
        """
        union two structs and return self

        Args:
            other (DynamicStruct): the other struct

        Returns:
            DynamicStruct: this struct
        """
        self._atoms.extend(other.atoms)
        self._bonds.extend(other.bonds)
        self._angles.extend(other.angles)
        self._dihedrals.extend(other.dihedrals)
        self._topology.union(other.topology)
        return self

    def clone(self):
        struct = DynamicStruct()
        struct.union(self)
        return struct
    
    def check_atom_alignment(self):

        # check atom
        atom_field = list(self.atoms[0].keys())
        for atom in self.atoms[1:]:
            assert atom_field == list(atom.keys()), ValueError("Atom fields are not aligned")

    def atom_keys(self):
        self.check_atom_alignment()
        return list(self.atoms[0].keys())
    
    def bond_keys(self):
        return list(self.bonds[0].keys())
    
    def angle_keys(self):
        return list(self.angles[0].keys())
    
    def to_struct(self):
        
        struct = Struct()
        
        atom_keys = self.atom_keys()
        for key in atom_keys:
            struct.atoms[key] = self.atoms[key]

        bond_keys = list(self.bonds[0].keys())
        for key in bond_keys:
            struct.bonds[key] = self.bonds[key]

        angle_keys = list(self.angles[0].keys())
        for key in angle_keys:
            struct.angles[key] = self.angles[key]

        return struct
    