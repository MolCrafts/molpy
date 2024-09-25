import numpy as np
from typing import Literal
from functools import reduce
from pathlib import Path
from typing import Iterable
from ..potential.base import Potential


class Style:

    def __init__(
        self,
        style: str,
        mixing=Literal["geometry", "arithmetic"] | None,
        *params,
        **named_params,
    ):
        self.params = list(params)
        self.named_params = named_params
        if isinstance(style, str):
            self.name = style
            self.calculator = None
        self.types: list[Type] = []
        self.mixing = mixing

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __contains__(self, name: str):
        for type_ in self.types:
            if type_.name == name:
                return True
        return False

    @property
    def n_types(self):
        return len(self.types)

    def get_params(self):
        raise NotImplementedError("get_params method must be implemented")

    def get_type(self, name: str):
        for type_ in self.types:
            if type_.name == name:
                return type_
        return None


class Type:

    def __init__(
        self, name: str | Potential, type_idx: tuple[int], *params, **named_params
    ):

        self.params = list(params)
        self.named_params = named_params

        assert all(isinstance(idx, (int | None)) for idx in type_idx), TypeError(
            f"type_idx must be a tuple of integers or None(to be defined later), but got {type_idx}"
        )
        self.name = name

        self.type_idx = type_idx

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __setitem__(self, key: str, value: float):
        self.named_params[key] = value

    def __getitem__(self, key: str):
        return self.named_params[key]


class AtomType(Type):

    def __init__(self, name: str, idx: int, *params, **named_params: dict):
        super().__init__(name, (idx,), params, **named_params)


class AtomStyle(Style):

    def def_type(self, name: str, type_idx: int, *params, **named_params: dict):
        at = AtomType(name, type_idx, *params, **named_params)
        self.types.append(at)
        return at


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

    def def_type(
        self,
        name: str = "",
        idx_i: int | None = None,
        idx_j: int | None = None,
        /,
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

    def get_params(self):

        index = []
        params = {}

        for type_ in self.types:
            index.append(type_.type_idx)
            for k, v in type_.named_params.items():
                if k not in params:
                    params[k] = []
                params[k].append(v)

        index = np.array(index)
        n_types = index.max() + 1
        flatten_params = {}
        for key in params:
            param_arr = np.zeros((n_types, n_types))
            param_arr[index[:, 0], index[:, 1]] = params[key]
            param_arr[index[:, 1], index[:, 0]] = params[key]
            flatten_params[key] = param_arr

        return flatten_params


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

    def def_type(
        self,
        name: str = "",
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        *params,
        **named_params,
    ):
        self.types.append(AngleType(name, idx_i, idx_j, idx_k, *params, **named_params))
        return self.types[-1]

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

    def def_type(
        self,
        name: str = "",
        idx_i: int | None = None,
        idx_j: int | None = None,
        idx_k: int | None = None,
        idx_l: int | None = None,
        /,
        *params,
        **named_params,
    ):
        self.types.append(
            DihedralType(name, idx_i, idx_j, idx_k, idx_l, *params, **named_params)
        )
        return self.types[-1]


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

    def def_type(
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
        return self.types[-1]


class PairType(Type):

    def __init__(self, id, i, j, *params, **named_params):
        super().__init__(id, (i, j), *params, **named_params)

    @property
    def idx_i(self):
        return self.type_idx[0]

    @property
    def idx_j(self):
        return self.type_idx[1]


class PairStyle(Style):

    def def_type(
        self, name: str, idx_i: int | None, idx_j: int | None, *params, **named_params
    ):
        self.types.append(PairType(name, idx_i, idx_j, *params, **named_params))
        return self.types[-1]

    def get_params(self):

        idx_i = []
        idx_j = []
        params = {}
        for pairtype in self.types:
            idx_i.append(pairtype.idx_i)
            idx_j.append(pairtype.idx_j)
            for k, v in pairtype.named_params.items():
                if k not in params:
                    params[k] = []
                params[k].append(v)

        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        n_types_i = idx_i.max() + 1
        n_types_j = idx_j.max() + 1

        flatten_params = {}
        for k in params:
            n_types = max(n_types_i, n_types_j)
            param_arr = np.zeros((n_types, n_types))
            param_arr[idx_i, idx_j] = params[k]
            param_arr[idx_j, idx_i] = params[k]
            if self.mixing == "arithmetic":
                param_arr = 0.5 * (param_arr + param_arr.T)

            elif self.mixing == "geometric":
                param_arr = np.sqrt(param_arr * param_arr.T)
            flatten_params[k] = param_arr

        return flatten_params


class ForceField:

    def __init__(self, name: str = ""):

        self.name = name

        self.unit = ""
        self.atomstyles: list[AtomStyle] = []
        self.bondstyles: list[BondStyle] = []
        self.pairstyles: list[PairStyle] = []
        self.anglestyles: list[AngleStyle] = []
        self.dihedralstyles: list[DihedralStyle] = []
        self.improperstyles: list[ImproperStyle] = []

    def __repr__(self) -> str:
        return f"<ForceField: {self.name}>"

    def __str__(self) -> str:
        detail = f"<ForceField: {self.name}"
        if self.n_atomstyles > 0:
            detail += (
                f"\nn_atomstyles: {self.n_atomstyles}, n_atomtypes: {self.n_atomtypes}"
            )
        if self.n_bondstyles > 0:
            detail += (
                f"\nn_bondstyles: {self.n_bondstyles}, n_bondtypes: {self.n_bondtypes}"
            )
        if self.n_pairstyles > 0:
            detail += (
                f"\nn_pairstyles: {self.n_pairstyles}, n_pairtypes: {self.n_pairtypes}"
            )
        if self.n_anglestyles > 0:
            detail += f"\nn_anglestyles: {self.n_anglestyles}, n_angletypes: {self.n_angletypes}"
        if self.n_dihedralstyles > 0:
            detail += f"\nn_dihedralstyles: {self.n_dihedralstyles}, n_dihedraltypes: {self.n_dihedraltypes}"
        if self.n_improperstyles > 0:
            detail += f"\nn_improperstyles: {self.n_improperstyles}, n_impropertypes: {self.n_impropertypes}"
        return detail + ">"

    def def_bondstyle(self, style: str, *params, **named_params):
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
    
    def get_improperstyle(self, style: str):
        for improperstyle in self.improperstyles:
            if improperstyle.name == style:
                return improperstyle
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

    def simplify(self):

        pass
