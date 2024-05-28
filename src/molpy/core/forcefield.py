import numpy as np
from typing import Literal
from functools import reduce
import molpy as mp
from molpy.core.space import Box
from molpy.core.struct import Struct
from molpy.potential.base import Potential


class Style(dict):

    def __init__(
        self,
        style: str | type[Potential],
        mixing=Literal["geometry", "arithmetic"] | None,
        **global_params,
    ):
        super().__init__(**global_params)
        if isinstance(style, str):
            self.name = style
            self.calculator = None
        elif issubclass(style, Potential):
            self.name = style.name
            self.calculator = style(**global_params)
        self.types: list[Type] = []
        self.mixing = mixing

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    @property
    def n_types(self):
        return len(self.types)
    
    def get_param(self, key: str):
        raise NotImplementedError("get_param method must be implemented")

    @property
    def params(self):
        return {
            field: self.get_param(field)
            for field in self.calculator.registered_params
        }

    def calc_struct(self, struct: Struct, output: dict):

        if self.calculator is None:
            raise ValueError("style must be a subclass of Potential")

        return self.calculator(struct, output, **self.params)


class Type(dict):

    def __init__(self, name: str, *type_idx: tuple[int], **props):
        super().__init__(**props)
        assert isinstance(name, str), TypeError("name must be a string")
        assert all(isinstance(idx, (int|None)) for idx in type_idx), TypeError(
            "type_idx must be a tuple of integers or None(to be defined later)"
        )
        self.name = name
        
        self.type_idx = type_idx

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class AtomType(Type):

    def __init__(self, name: str, idx: int, **props: dict):
        super().__init__(name, idx, **props)


class AtomStyle(Style):

    def def_atomtype(self, name: str, type_idx: int, **props: dict):
        at = AtomType(name, type_idx, **props)
        self.types.append(at)
        return at

    def get_atomtype(self, name: str):
        for atomtype in self.types:
            if atomtype.name == name:
                return atomtype
        return None


class BondType(Type):

    def __init__(self, name: str, idx_i: int | None, idx_j: int | None, **params):
        super().__init__(name, idx_i, idx_j, **params)

    @property
    def idx_i(self):
        return self.type_idx[0]

    @property
    def idx_j(self):
        return self.type_idx[1]


class BondStyle(Style):

    def def_bondtype(
        self, idx_i: int | None=None, idx_j: int | None=None, /, name: str = "", **params
    ) -> BondType:
        """
        define bond type

        Args:
            id (int): bond type id
            idx_i (int | None): atom type index i
            idx_j (int | None): atom type index j

        Returns:
            BondType: defined bond type
        """
        self.types.append(BondType(name, idx_i, idx_j, **params))
        return self.types[-1]
    
    def get_param(self, key:str):

        types = []
        params = []
        for type_ in self.types:
            types.append( type_.type_idx )
            params.append(type_[key])

        n_types = np.max(types) + 1
        param_arr = np.zeros([n_types, n_types], dtype=float)
        for type_, param in zip(types, params):
            param_arr[type_[0], type_[1]] = param
            param_arr[type_[1], type_[0]] = param

        return param_arr

class AngleType(Type):

    def __init__(
        self,
        name: str,
        idx_i: int | None,
        idx_j: int | None,
        idx_k: int | None,
        **params,
    ):
        super().__init__(name, idx_i, idx_j, idx_k, **params)

    @property
    def idx_i(self):
        return self.type_idx[0]

    @property
    def idx_j(self):
        return self.type_idx[1]
    
    @property
    def idx_k(self):
        return self.type_idx[2]



class DihedralType(Type):

    def __init__(
        self,
        name: str,
        idx_i: int | None,
        idx_j: int | None,
        idx_k: int | None,
        idx_l: int | None,
        *args,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k
        self.idx_l = idx_l


class AngleStyle(Style):

    def def_angletype(
        self,
        idx_i: int | None=None,
        idx_j: int | None=None,
        idx_k: int | None=None,
        name: str = "",
        *args,
        **kwargs,
    ):
        self.types.append(AngleType(name, idx_i, idx_j, idx_k, **kwargs))

    def get_param(self, key: str):
        idx_i = []
        idx_j = []
        idx_k = []
        params = []
        for angletype in self.types:
            idx_i.append(angletype.idx_i)
            idx_j.append(angletype.idx_j)
            idx_k.append(angletype.idx_k)
            params.append(angletype[key])

        n_types_i = np.max(idx_i) + 1
        n_types_j = np.max(idx_j) + 1
        n_types_k = np.max(idx_k) + 1
        n_types = max(n_types_i, n_types_j, n_types_k)
        param_arr = np.zeros((n_types, n_types, n_types))
        for i, j, k, param in zip(idx_i, idx_j, idx_k, params):
            param_arr[i, j, k] = param
            param_arr[k, j, i] = param

        return param_arr


class DihedralStyle(Style):

    def def_dihedraltype(
        self,
        name: str="",
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        idx_l: int | None = None,
        *args,
        **kwargs,
    ):
        self.types.append(DihedralType(name, idx_i, idx_j, idx_k, idx_l, **kwargs))


class ImproperType(Type):

    def __init__(
        self,
        name: str,
        idx_i: int | None,
        idx_j: int | None,
        idx_k: int | None,
        idx_l: int | None,
        *args,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.idx_i = idx_i
        self.idx_j = idx_j
        self.idx_k = idx_k
        self.idx_l = idx_l


class ImproperStyle(Style):

    def def_impropertype(
        self,
        name: str="",
        idx_i: int | None=None,
        idx_j: int | None=None,
        idx_k: int | None=None,
        idx_l: int | None=None,
        *args,
        **kwargs,
    ):
        self.types.append(ImproperType(name, idx_i, idx_j, idx_k, idx_l, **kwargs))


class PairType(Type):

    def __init__(self, id, i, j, **kwargs):
        super().__init__(id, **kwargs)
        self.i = i
        self.j = j


class PairStyle(Style):

    def def_pairtype(self, name: str, idx_i, idx_j, **kwargs):
        self.types.append(PairType(name, idx_i, idx_j, **kwargs))

    def get_pairtype_params(self, key: str):
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

        if self.mix == "geometry":
            for i in range(n_types):
                for j in range(n_types):
                    temp = np.sqrt(param_arr[i, i] * param_arr[j, j])
                    param_arr[i, j] = temp
                    param_arr[j, i] = temp
        elif self.mix == "arithmetic":
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

    def def_bondstyle(self, style: str | type[Potential], **global_params):
        bondstyle = BondStyle(style, **global_params)
        self.bondstyles.append(bondstyle)
        return bondstyle

    def def_pairstyle(self, style: str, **kwargs):
        pairstyle = PairStyle(style, **kwargs)
        self.pairstyles.append(pairstyle)
        return pairstyle

    def def_atomstyle(self, style: str, **props: dict):
        atomstyle = AtomStyle(style, **props)
        self.atomstyles.append(atomstyle)
        return atomstyle

    def def_anglestyle(self, style: str, **kwargs):
        anglestyle = AngleStyle(style, **kwargs)
        self.anglestyles.append(anglestyle)
        return anglestyle

    def def_dihedralstyle(self, style: str, **kwargs):
        dihedralstyle = DihedralStyle(style, **kwargs)
        self.dihedralstyles.append(dihedralstyle)
        return dihedralstyle

    def def_improperstyle(self, style: str, **kwargs):
        improperstyle = ImproperStyle(style, **kwargs)
        self.improperstyles.append(improperstyle)
        return improperstyle

    def get_pairstyle(self, style: str):
        for pairstyle in self.pairstyles:
            if pairstyle.style == style:
                return pairstyle
        return None

    def get_bondstyle(self, style: str):
        for bondstyle in self.bondstyles:
            if bondstyle.style == style:
                return bondstyle
        return None

    def get_atomstyle(self, style: str):
        for atomstyle in self.atomstyles:
            if atomstyle.style == style:
                return atomstyle
        return None

    def get_anglestyle(self, style: str):
        for anglestyle in self.anglestyles:
            if anglestyle.style == style:
                return anglestyle
        return None

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

    def calc_struct(self, struct: Struct, output:dict={})->dict:
        
        struct, output = self.calc_bond(struct, output)
        return struct, output

    def calc_bond(self, struct, output:dict={}): 

        for bs in self.bondstyles:
            struct, output = bs.calc_struct(struct, output)

        return struct, output
    
    def get_calculator(self):

        pot = []
        for bs in self.bondstyles:
            if bs.calculator:
                pot.append(bs.calculator)
        return PotentialSeq(*pot, name=self.name)
                
