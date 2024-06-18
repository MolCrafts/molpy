import numpy as np
from typing import Literal
from functools import reduce
import molpy as mp
from molpy.core.space import Box
from molpy.core.struct import Struct
from molpy.potential.base import Potential
from pathlib import Path
from typing import Iterable


class Style:

    def __init__(
        self,
        style: str | type[Potential],
        mixing=Literal["geometry", "arithmetic"] | None,
        *params,
        **named_params,
    ):
        self.params = list(params)
        self.named_params = named_params
        if isinstance(style, str):
            self.name = style
            self.calculator = None
        elif issubclass(style, Potential):
            self.name = style.name
            self.calculator = style(**named_params)
        self.types: list[Type] = []
        self.mixing = mixing

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    @property
    def n_types(self):
        return len(self.types)

    def get_param(self, key: str):
        raise NotImplementedError("get_param method must be implemented")

    def calc_struct(self, struct: Struct, output: dict):

        if self.calculator is None:
            raise ValueError("style must be a subclass of Potential")

        return self.calculator(struct, output, **self.params)


class Type:

    def __init__(self, name: str, type_idx: tuple[int], *params, **named_params):

        self.params = list(params)
        self.named_params = named_params

        assert isinstance(name, str), TypeError("name must be a string")
        assert all(isinstance(idx, (int | None)) for idx in type_idx), TypeError(
            "type_idx must be a tuple of integers or None(to be defined later)"
        )
        self.name = name

        self.type_idx = type_idx

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class AtomType(Type):

    def __init__(self, name: str, idx: int, *params, **named_params: dict):
        super().__init__(name, (idx,), params, **named_params)


class AtomStyle(Style):

    def def_atomtype(self, name: str, type_idx: int, *params, **named_params: dict):
        at = AtomType(name, type_idx, *params, **named_params)
        self.types.append(at)
        return at

    def get_atomtype(self, name: str):
        for atomtype in self.types:
            if atomtype.name == name:
                return atomtype
        return None


class BondType(Type):

    def __init__(
        self, name: str, idx_i: int | None, idx_j: int | None, *params, **named_params
    ):
        super().__init__(name, (idx_i, idx_j), *params, **named_params)

    @property
    def idx_i(self):
        return self.type_idx[0]

    @property
    def idx_j(self):
        return self.type_idx[1]


class BondStyle(Style):

    def def_bondtype(
        self,
        idx_i: int | None = None,
        idx_j: int | None = None,
        /,
        name: str = "",
        *params,
        **named_params,
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
        self.types.append(BondType(name, idx_i, idx_j, *params, **named_params))
        return self.types[-1]

    def get_param(self, key: str):

        types = []
        params = []
        for type_ in self.types:
            types.append(type_.type_idx)
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
        *params,
        **named_params,
    ):
        super().__init__(name, (idx_i, idx_j, idx_k), *params, **named_params)

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
        *params,
        **named_params,
    ):
        super().__init__(name, (idx_i, idx_j, idx_k, idx_l), *params, **named_params)


class AngleStyle(Style):

    def def_angletype(
        self,
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        name: str = "",
        *params,
        **named_params,
    ):
        self.types.append(AngleType(name, idx_i, idx_j, idx_k, *params, **named_params))

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
        name: str = "",
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        idx_l: int | None = None,
        *params,
        **named_params,
    ):
        self.types.append(
            DihedralType(name, idx_i, idx_j, idx_k, idx_l, *params, **named_params)
        )


class ImproperType(Type):

    def __init__(
        self,
        name: str,
        idx_i: int | None,
        idx_j: int | None,
        idx_k: int | None,
        idx_l: int | None,
        *params,
        **named_params,
    ):
        super().__init__(name, (idx_i, idx_j, idx_k, idx_l), *params, **named_params)


class ImproperStyle(Style):

    def def_impropertype(
        self,
        name: str = "",
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        idx_l: int | None = None,
        *params,
        **named_params,
    ):
        self.types.append(
            ImproperType(name, idx_i, idx_j, idx_k, idx_l, *params, **named_params)
        )


class PairType(Type):

    def __init__(self, id, i, j, *params, **named_params):
        super().__init__(id, (i, j), *params, **named_params)


class PairStyle(Style):

    def def_pairtype(
        self, name: str, idx_i: int | None, idx_j: int | None, *params, **named_params
    ):
        self.types.append(PairType(name, idx_i, idx_j, *params, **named_params))

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

    def __init__(self, name:str=""):

        self.name = name

        self.unit = ""
        self.atomstyles = []
        self.bondstyles = []
        self.pairstyles = []
        self.anglestyles = []
        self.dihedralstyles = []
        self.improperstyles = []

    # def read_lammps(self, fpaths: list[str | Path]):
    #     from molpy.io.forcefield import LAMMPSForceFieldReader
    #     LAMMPSForceFieldReader(fpaths, self).read()

    # def write_lammps(self, fpath: str | Path):
    #     from molpy.io.forcefield import LAMMPSForceFieldWriter
    #     LAMMPSForceFieldWriter(fpath, self).write() 

    def __repr__(self) -> str:
        return f"<ForceField: {self.name}>"
    
    def __str__(self) -> str:
        detail = f"<ForceField: {self.name}"
        if self.n_atomstyles > 0:
            detail += f"\nn_atomstyles: {self.n_atomstyles}, n_atomtypes: {self.n_atomtypes}"
        if self.n_bondstyles > 0:
            detail += f"\nn_bondstyles: {self.n_bondstyles}, n_bondtypes: {self.n_bondtypes}"
        if self.n_pairstyles > 0:
            detail += f"\nn_pairstyles: {self.n_pairstyles}, n_pairtypes: {self.n_pairtypes}"
        if self.n_anglestyles > 0:
            detail += f"\nn_anglestyles: {self.n_anglestyles}, n_angletypes: {self.n_angletypes}"
        if self.n_dihedralstyles > 0:
            detail += f"\nn_dihedralstyles: {self.n_dihedralstyles}, n_dihedraltypes: {self.n_dihedraltypes}"
        if self.n_improperstyles > 0:
            detail += f"\nn_improperstyles: {self.n_improperstyles}, n_impropertypes: {self.n_impropertypes}"
        return detail + ">"
    
    def def_bondstyle(self, style: str | type[Potential], *params, **named_params):
        bondstyle = BondStyle(style, *params, **named_params)
        self.bondstyles.append(bondstyle)
        return bondstyle

    def def_pairstyle(self, style: str, *params, **named_params):
        pairstyle = PairStyle(style, *params, **named_params)
        self.pairstyles.append(pairstyle)
        return pairstyle

    def def_atomstyle(self, style: str, **params: dict):
        atomstyle = AtomStyle(style, **params)
        self.atomstyles.append(atomstyle)
        return atomstyle

    def def_anglestyle(self, style: str, *params, **named_params):
        anglestyle = AngleStyle(style, *params, **named_params)
        self.anglestyles.append(anglestyle)
        return anglestyle

    def def_dihedralstyle(self, style: str, *params, **named_params):
        dihedralstyle = DihedralStyle(style, *params, **named_params)
        self.dihedralstyles.append(dihedralstyle)
        return dihedralstyle

    def def_improperstyle(self, style: str, *params, **named_params):
        improperstyle = ImproperStyle(style, *params, **named_params)
        self.improperstyles.append(improperstyle)
        return improperstyle

    def get_pairstyle(self, style: str):
        for pairstyle in self.pairstyles:
            if pairstyle.name == style:
                return pairstyle
        return None

    def get_bondstyle(self, style: str):
        for bondstyle in self.bondstyles:
            if bondstyle.name == style:
                return bondstyle
        return None

    def get_atomstyle(self, style: str):
        for atomstyle in self.atomstyles:
            if atomstyle.name == style:
                return atomstyle
        return None

    def get_anglestyle(self, style: str):
        for anglestyle in self.anglestyles:
            if anglestyle.name == style:
                return anglestyle
        return None

    def get_dihedralstyle(self, style: str):
        for dihedralstyle in self.dihedralstyles:
            if dihedralstyle.name == style:
                return dihedralstyle
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
    
    @property
    def atomtypes(self):
        return reduce(lambda x, y: x + y.types, self.atomstyles, [])
    
    @property
    def bondtypes(self):
        return reduce(lambda x, y: x + y.types, self.bondstyles, [])
    
    @property
    def angletypes(self):
        return reduce(lambda x, y: x + y.types, self.anglestyles, [])
    
    @property
    def dihedraltypes(self):
        return reduce(lambda x, y: x + y.types, self.dihedralstyles, [])
    
    @property
    def impropertypes(self):
        return reduce(lambda x, y: x + y.types, self.improperstyles, [])
    
    @property
    def pairtypes(self):
        return reduce(lambda x, y: x + y.types, self.pairstyles, [])

    def append(self, forcefield: "ForceField"):

        if self.unit:
            assert self.unit == forcefield.unit, ValueError("unit must be the same")
        self.unit = forcefield.unit


        for pairstyle in forcefield.pairstyles:
            if self.get_pairstyle(pairstyle.name) is None:
                self.pairstyles.append(pairstyle)
            else:
                self_pairstyle = self.get_pairstyle(pairstyle.name)
                for pairtype in pairstyle.types:
                    self_pairstyle.types.append(pairtype)

        for bondstyle in forcefield.bondstyles:
            if self.get_bondstyle(bondstyle.name) is None:
                self.bondstyles.append(bondstyle)
            else:
                self_bondstyle = self.get_bondstyle(bondstyle.name)
                for bondtype in bondstyle.types:
                    self_bondstyle.types.append(bondtype)
        
        for atomstyle in forcefield.atomstyles:
            if self.get_atomstyle(atomstyle.name) is None:
                self.atomstyles.append(atomstyle)
            else:
                self_atomstyle = self.get_atomstyle(atomstyle.name)
                for atomtype in atomstyle.types:
                    self_atomstyle.types.append(atomtype)

        for anglestyle in forcefield.anglestyles:
            if self.get_anglestyle(anglestyle.name) is None:
                self.anglestyles.append(anglestyle)
            else:
                self_anglestyle = self.get_anglestyle(anglestyle.name)
                for angletype in anglestyle.types:
                    self_anglestyle.types.append(angletype)

        for dihedralstyle in forcefield.dihedralstyles:
            if self.get_dihedralstyle(dihedralstyle.name) is None:
                self.dihedralstyles.append(dihedralstyle)
            else:
                self_dihedralstyle = self.get_dihedralstyle(dihedralstyle.name)
                for dihedraltype in dihedralstyle.types:
                    self_dihedralstyle.types.append(dihedraltype)

        for improperstyle in forcefield.improperstyles:
            if self.get_improperstyle(improperstyle.name) is None:
                self.improperstyles.append(improperstyle)
            else:
                self_improperstyle = self.get_improperstyle(improperstyle.name)
                for impropertype in improperstyle.types:
                    self_improperstyle.types.append(impropertype)

    def extend(self, forcefields: Iterable["ForceField"]):
        for ff in forcefields:
            self.append(ff)

    def calc_struct(self, struct: Struct, output: dict = {}) -> dict:

        struct, output = self.calc_bond(struct, output)
        return struct, output

    def calc_bond(self, struct, output: dict = {}):

        for bs in self.bondstyles:
            struct, output = bs.calc_struct(struct, output)

        return struct, output

    # def get_calculator(self):

    #     pot = []
    #     for bs in self.bondstyles:
    #         if bs.calculator:
    #             pot.append(bs.calculator)
    #     return PotentialSeq(*pot, name=self.name)
