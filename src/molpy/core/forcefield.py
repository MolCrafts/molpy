import numpy as np
from functools import reduce
from typing import Iterable

class Type(dict):

    def __init__(
        self, id: int, name: str = "", kw_params: dict = {}, order_params: list = []
    ):
        super().__init__(kw_params)
        self.id = id
        self.name = name
        self['order_params'] = order_params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.id}>"


class Style(dict):

    def __init__(
        self,
        name: str,
        *params,
        **named_params,
    ):
        super().__init__(**named_params)
        self.name = name
        self.types: dict[int, Type] = {}
        self['params'] = params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


    @property
    def n_types(self):
        return len(self.types)

    def get_type(self, name: str) -> Type | None:
        for type_ in self.types:
            if type_.name == name:
                return type_
        return None

    def __contains__(self, id: int):
        return id in self.types
    
    def __getitem__(self, id: int):
        return self.types[id]
    
    def __eq__(self, other: "Style"):
        return self.name == other.name
    
    def remap_id(self):
        new_types = {}
        for i, t in self.types.items():
            new_types[i] = t
            t.id = i
        self.types = new_types

    def get_type_by_name(self, name: str) -> Type | None:
        for typ in self.types.values():
            if typ.name == name:
                return typ
        return None

    def merge(self, other: "Style"):
        self.update(other)
        max_id = max(self.types.keys())
        for typ in other.types.values():
            if self.get_type_by_name(typ.name) is None:
                typ.id += max_id
                self.types[typ.id] = typ

class AtomType(Type):

    def __init__(self, id:int, name:str = "", kw_params: dict = {}, order_params: list = []):
        super().__init__(id, name, kw_params, order_params)


class AtomStyle(Style):

    def def_type(self, id:int, name:str = "", kw_params: dict = {}, order_params: list = []):
        at = AtomType(id, name, kw_params, order_params)
        self.types[id] = at
        return at


class BondType(Type):

    def __init__(
        self, id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []
    ):
        super().__init__(id, name, kw_params, order_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype


class BondStyle(Style):

    def def_type(
        self,
        id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []
    ) -> BondType:
        bt = BondType(id, itomtype, jtomtype, name, kw_params, order_params)
        self.types[id] = bt
        return bt

class AngleType(Type):

    def __init__(self, id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []):
        super().__init__(id, name, kw_params, order_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype


class DihedralType(Type):

    def __init__(self, id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, ltomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []):
        super().__init__(id, name, kw_params, order_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype
        self.ltomtype = ltomtype


class AngleStyle(Style):

    def def_type(
        self,
        id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []
    ) -> AngleType:
        at = AngleType(id, itomtype, jtomtype, ktomtype, name, kw_params, order_params)
        self.types[id] = at
        return at

class DihedralStyle(Style):

    def def_type(
        self,
        id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, ltomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []
    ) -> DihedralType:
        dt = DihedralType(id, itomtype, jtomtype, ktomtype, ltomtype, name, kw_params, order_params)
        self.types[id] = dt
        return dt


class ImproperType(Type):

    def __init__(self, id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, ltomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []):
        super().__init__(id, name, kw_params, order_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype
        self.ltomtype = ltomtype


class ImproperStyle(Style):

    def def_type(
        self,
        id:int, itomtype: AtomType|None = None, jtomtype: AtomType|None = None, ktomtype: AtomType|None = None, ltomtype: AtomType|None = None, name:str = "", kw_params: dict = {}, order_params: list = []
    ) -> ImproperType:
        it = ImproperType(id, itomtype, jtomtype, ktomtype, ltomtype, name, kw_params, order_params)
        self.types[id] = it
        return it


class PairType(Type):

    def __init__(self, id: int, itomtype: int | None, jtomtype: int | None, name: str = "", kw_params: dict = {}, order_params: list = []):
        super().__init__(id, name, kw_params, order_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype

class PairStyle(Style):

    def def_type(
        self,
        id: int, itomtype: int | None, jtomtype: int | None, name: str = "", kw_params: dict = {}, order_params: list = []
    ) -> PairType:
        pt = PairType(id, itomtype, jtomtype, name, kw_params, order_params)
        self.types[id] = pt
        return pt

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
    
    def get_atomtypes(self):
        return reduce(lambda x, y: x + y.types, self.atomstyles, [])
    
    def get_bondtypes(self):
        return reduce(lambda x, y: x + y.types, self.bondstyles, [])
    
    def get_angletypes(self):
        return reduce(lambda x, y: x + y.types, self.anglestyles, [])
    
    def get_dihedraltypes(self):
        return reduce(lambda x, y: x + y.types, self.dihedralstyles, [])
    
    def get_impropertypes(self):
        return reduce(lambda x, y: x + y.types, self.improperstyles, [])

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

    def merge(self, other: "ForceField"):

        if self.unit:
            assert self.unit == other.unit, ValueError("unit must be the same")
        self.unit = other.unit

        def _merge(styles, other_styles):
            for style in other_styles:
                this_style = None
                for s in styles:
                    if s.name == style.name:
                        this_style = s
                        break
                if this_style:
                    this_style.merge(style)
                else:
                    this_style = style
                this_style = None

        _merge(self.atomstyles, other.atomstyles)
        _merge(self.bondstyles, other.bondstyles)
        _merge(self.pairstyles, other.pairstyles)
        _merge(self.anglestyles, other.anglestyles)
        _merge(self.dihedralstyles, other.dihedralstyles)
        _merge(self.improperstyles, other.improperstyles)


        return self

    def __iadd__(self, forcefield: "ForceField"):
        self.append(forcefield)
        return self
    
    def __add__(self, forcefield: "ForceField"):
        new_forcefield = ForceField()
        new_forcefield.merge(self)
        new_forcefield.merge(forcefield)
        return new_forcefield
    

    def extend(self, forcefields: Iterable["ForceField"]):
        for ff in forcefields:
            self.append(ff)

    def simplify(self):

        pass
