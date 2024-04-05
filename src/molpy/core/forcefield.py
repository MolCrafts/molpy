import numpy as np
from typing import Literal
from functools import reduce

class Style(dict):

    def __init__(self, name:str, mix=Literal['geometry', 'arithmetic']|None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['name'] = name
        self.types = []
        self.mix = mix

    @property
    def n_types(self):
        return len(self.types)

class Type(dict):
    
    def __init__(self, name:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self['name'] = name

class AtomType(Type):

    def __init__(self, name:str, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

class AtomStyle(Style):
    
    def def_atomtype(self, name:str, *args, **kwargs):
        at = AtomType(name, *args, **kwargs)
        self.types.append(at)
        return at

    def get_atomtype(self, name:str):
        for atomtype in self.types:
            if atomtype['name'] == name:
                return atomtype
        return None

class BondType(Type):

    def __init__(self, name:str, idx_i:int|None, idx_j:int|None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j

class BondStyle(Style):
    
    def def_bondtype(self, name:str, idx_i:int|None, idx_j:int|None, *args, **kwargs):
        """
        define bond type

        Args:
            id (int): bond type id
            idx_i (int | None): atom type id i
            idx_j (int | None): atom type id j

        Returns:
            BondType: defined bond type
        """
        bondtype = BondType(name, idx_i, idx_j, *args, **kwargs)
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

    def __init__(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k

class DihedralType(Type):

    def __init__(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k
        self.idx_l = idx_l

class AngleStyle(Style):
    
    def def_angletype(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, *args, **kwargs):
        self.types.append(AngleType(name, idx_i, idx_j, idx_k, *args, **kwargs))

class DihedralStyle(Style):
    
    def def_dihedraltype(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        self.types.append(DihedralType(name, idx_i, idx_j, idx_k, idx_l, *args, **kwargs))

class ImproperType(Type):

    def __init__(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k
        self.idx_l = idx_l

class ImproperStyle(Style):

    def def_impropertype(self, name:str, idx_i:int|None, idx_j:int|None, idx_k:int|None, idx_l:int|None, *args, **kwargs):
        self.types.append(ImproperType(name, idx_i, idx_j, idx_k, idx_l, *args, **kwargs))

class PairType(Type):
    
    def __init__(self, id, i, j, *args, **kwargs):
        super().__init__(id, *args, **kwargs)
        self.i = i
        self.j = j

class PairStyle(Style):
    
    def def_pairtype(self, name:str, idx_i, idx_j, *args, **kwargs):
        self.types.append(PairType(name, idx_i, idx_j, *args, **kwargs))

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

        self.atomstyles = []
        self.bondstyles = []
        self.pairstyles = []
        self.anglestyles = []
        self.dihedralstyles = []
        self.improperstyles = []

    def def_bondstyle(self, style:str, *args, **kwargs):
        bondstyle = BondStyle(style, *args, **kwargs)
        self.bondstyles.append(bondstyle)
        return bondstyle

    def def_pairstyle(self, style:str, *args, **kwargs):
        pairstyle = PairStyle(style, *args, **kwargs)
        self.pairstyles.append(pairstyle)
        return pairstyle

    def def_atomstyle(self, style:str, *args, **kwargs):
        atomstyle = AtomStyle(style, *args, **kwargs)
        self.atomstyles.append(atomstyle)
        return atomstyle
    
    def def_anglestyle(self, style:str, *args, **kwargs):
        anglestyle = AngleStyle(style, *args, **kwargs)
        self.anglestyles.append(anglestyle)
        return anglestyle
    
    def def_dihedralstyle(self, style:str, *args, **kwargs):
        dihedralstyle = DihedralStyle(style, *args, **kwargs)
        self.dihedralstyles.append(dihedralstyle)
        return dihedralstyle
    
    def def_improperstyle(self, style:str, *args, **kwargs):
        improperstyle = ImproperStyle(style, *args, **kwargs)
        self.improperstyles.append(improperstyle)
        return improperstyle

    @property
    def n_atomstyles(self):
        return len(self.atomstyles)
    
    @property
    def n_bondstyles(self):
        return len(self.bondstyles)
    
    @property
    def n_pairstyles(self):
        return len(self.pairstyles)
    
    @property
    def n_anglestyles(self):
        return len(self.anglestyles)
    
    @property
    def n_dihedralstyles(self):
        return len(self.dihedralstyles)
    
    @property
    def n_improperstyles(self):
        return len(self.improperstyles)
    
    @property
    def n_atomtypes(self):
        return reduce(lambda x, y: x + y.n_types, self.atomstyles, 0)
    
    @property
    def n_bondtypes(self):
        return reduce(lambda x, y: x + y.n_types, self.bondstyles, 0)
    
    @property
    def n_angletypes(self):
        return reduce(lambda x, y: x + y.n_types, self.anglestyles, 0)
    
    @property
    def n_dihedraltypes(self):
        return reduce(lambda x, y: x + y.n_types, self.dihedralstyles, 0)
    
    @property
    def n_impropertypes(self):
        return reduce(lambda x, y: x + y.n_types, self.improperstyles, 0)
    
    @property
    def n_pairtypes(self):
        return reduce(lambda x, y: x + y.n_types, self.pairstyles, 0)
    