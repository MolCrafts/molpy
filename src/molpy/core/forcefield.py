import numpy as np
from molpy.io.forcefield import load_forcefield
from typing import Literal

class Style(dict):

    def __init__(self, name:str, mix=Literal['geometry', 'arithmetic']|None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.types = []
        self.mix = mix

    @property
    def n_types(self):
        return len(self.types)

class Type(dict):
    
    def __init__(self, id:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = id

class AtomType(Type):

    def __init__(self, id:int, *args, **kwargs):
        super().__init__(id, *args, **kwargs)

class AtomStyle(Style):
    
    def def_atom_type(self, id:int, *args, **kwargs):
        self.types.append(AtomType(id, *args, **kwargs))

class BondType(Type):

    def __init__(self, id:int, idx_i:int|None, idx_j:int|None, *args, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j

class BondStyle(Style):
    
    def def_bond_type(self, id:int, idx_i:int|None, idx_j:int|None, *args, **kwargs):
        """
        define bond type

        Args:
            id (int): bond type id
            idx_i (int | None): atom type id i
            idx_j (int | None): atom type id j

        Returns:
            BondType: defined bond type
        """
        bondtype = BondType(id, idx_i, idx_j, *args, **kwargs)
        self.types.append(bondtype)
        return bondtype

    def get_bondtype_params(self, key:str):
        idx_i = []
        idx_j = []
        params = []
        for bondtype in self.types:
            idx_i.append(bondtype.idx_i)
            idx_j.append(bondtype.idx_j)
            params.append(bondtype[key])

        n_types_i = np.max(idx_i) + 1
        n_types_j = np.max(idx_j) + 1
        n_types = max(n_types_i, n_types_j)
        param_arr = np.zeros((n_types, n_types))
        for i, j, param in zip(idx_i, idx_j, params):
            param_arr[i, j] = param
            param_arr[j, i] = param

        return param_arr

class AngleType(Type):

    def __init__(self, id:int, idx_i:int|None, idx_j:int|None, idx_k:int|None, *args, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k

class DihedralType(Type):

    def __init__(self, id:int, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k
        self.idx_l = idx_l

class AngleStyle(Style):
    
    def def_angle_type(self, id:int, idx_i:int|None, idx_j:int|None, idx_k:int|None, *args, **kwargs):
        self.types.append(AngleType(id, idx_i, idx_j, idx_k, *args, **kwargs))

class DihedralStyle(Style):
    
    def def_dihedral_type(self, id:int, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        self.types.append(DihedralType(id, idx_i, idx_j, idx_k, idx_l, *args, **kwargs))

class PairType(Type):
    
    def __init__(self, id, i, j, *args, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.i = i
        self.j = j

class PairStyle(Style):
    
    def def_pairtype(self, id:int, idx_i, idx_j, *args, **kwargs):
        self.types.append(PairType(id, idx_i, idx_j, *args, **kwargs))

    def get_pairtype_params(self, key:str):
        idx_i = []
        idx_j = []
        params = []
        for pairtype in self.types:
            idx_i.append(pairtype.i)
            idx_j.append(pairtype.j)
            params.append(pairtype[key])

        n_types_i = np.max(idx_i) + 1
        n_types_j = np.max(idx_j) + 1
        n_types = max(n_types_i, n_types_j)
        param_arr = np.zeros((n_types, n_types))
        for i, j, param in zip(idx_i, idx_j, params):
            param_arr[i, j] = param
            param_arr[j, i] = param
    
        if self.mix == 'geometry':        
            for i in range(n_types):
                for j in range(n_types):
                    temp = np.sqrt(param_arr[i, i] * param_arr[j, j])
                    param_arr[i, j] = temp
                    param_arr[j, i] = temp
        elif self.mix == 'arithmetic':
            for i in range(n_types):
                for j in range(n_types):
                    temp = 0.5 * (param_arr[i, i] + param_arr[j, j])
                    param_arr[i, j] = temp
                    param_arr[j, i] = temp

        return param_arr

class ForceField:

    def __init__(self):

        self.atom_styles = []
        self.bond_styles = []
        self.pair_styles = []

    @classmethod
    def from_file(cls, filename:str|list[str]):
        return load_forcefield(filename, forcefield=cls(), format='lammps')

    def def_bondstyle(self, style:str, *args, **kwargs):
        bondstyle = BondStyle(style, *args, **kwargs)
        self.bond_styles.append(bondstyle)
        return bondstyle

    def def_pairstyle(self, style:str, *args, **kwargs):
        pairstyle = PairStyle(style, *args, **kwargs)
        self.pair_styles.append(pairstyle)
        return pairstyle

    def def_atomstyle(self, style:str, *args, **kwargs):
        atomstyle = AtomStyle(style, *args, **kwargs)
        self.atom_styles.append(atomstyle)
        return atomstyle

    @property
    def n_atom_styles(self):
        return len(self.atom_styles)
    
    @property
    def n_bond_styles(self):
        return len(self.bond_styles)
    
    @property
    def n_pair_styles(self):
        return len(self.pair_styles)
    
    @property
    def n_angle_styles(self):
        return len(self.angle_styles)
    
    @property
    def n_dihedral_styles(self):
        return len(self.dihedral_styles)
    
    @property
    def n_improper_styles(self):
        return len(self.improper_styles)