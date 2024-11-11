import molpy as mp
from functools import reduce
from typing import Callable
from collections import defaultdict

class Type(dict):

    def __init__(self, name: str, *order_params, **kw_params):
        super().__init__(kw_params)
        self.name = name
        self.order_params = order_params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __eq__(self, other: "Type"):
        return self.name == other.name


class Style(dict):

    def __init__(self, name: str, *order_params, **kw_params):
        super().__init__(kw_params)
        self.name = name
        self.types: dict[str, Type] = {}
        self.order_params = order_params

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    @property
    def n_types(self):
        return len(self.types)

    def __eq__(self, other: "Style"):
        return self.name == other.name

    def get_by(self, condition: Callable, default=None) -> Type:
        for t in self.types.values():
            if condition(t):
                return t
        return default

    def get_all_by(self, condition: Callable):
        return [t for t in self.types.values() if condition(t)]

    def merge(self, other: "Style"):
        self.update(other)  # merge params
        self.types.update(other.types)

    def to_dict(self):
        return dict()


class AtomType(Type):

    def __init__(self, name: str, *order_params, **kw_params):
        super().__init__(name, *order_params, **kw_params)


class BondType(Type):

    def __init__(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ):
        super().__init__(name, *order_params, **kw_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype

    @property
    def atomtypes(self):
        return [self.itomtype, self.jtomtype]


class AngleType(Type):

    def __init__(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ):
        super().__init__(name, *order_params, **kw_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype

    @property
    def atomtypes(self):
        return [self.itomtype, self.jtomtype, self.ktomtype]


class ImproperType(Type):

    def __init__(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        ltomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ):
        super().__init__(name, *order_params, **kw_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype
        self.ltomtype = ltomtype

    @property
    def atomtypes(self):
        return [self.itomtype, self.jtomtype, self.ktomtype, self.ltomtype]


class DihedralType(Type):

    def __init__(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        ltomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ):
        super().__init__(name, *order_params, **kw_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype
        self.ktomtype = ktomtype
        self.ltomtype = ltomtype

    @property
    def atomtypes(self):
        return [self.itomtype, self.jtomtype, self.ktomtype, self.ltomtype]


class AtomStyle(Style):

    def __init__(self, name: str, *order_params, **kw_params):
        super().__init__(name, kw_params, order_params)
        self.classes = defaultdict(set)

    def def_type(self, name: str, class_=None, *order_params, **kw_params) -> AtomType:
        at = AtomType(name, *order_params, **kw_params)
        self.types[name] = at
        if class_:
            self.classes[name].add(class_)
        return at
    
    def get_class(self, class_: str) -> list[AtomType]:
        return self.classes.get(class_, [])


class BondStyle(Style):

    def def_type(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ) -> BondType:
        bt = BondType(name, itomtype, jtomtype, *order_params, **kw_params)
        self.types[name] = bt
        return bt


class AngleStyle(Style):

    def def_type(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ) -> AngleType:
        at = AngleType(name, itomtype, jtomtype, ktomtype, *order_params, **kw_params)
        self.types[name] = at
        return at


class DihedralStyle(Style):

    def def_type(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        ltomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ) -> DihedralType:
        dt = DihedralType(
            name, itomtype, jtomtype, ktomtype, ltomtype, *order_params, **kw_params
        )
        self.types[name] = dt
        return dt


class ImproperStyle(Style):

    def def_type(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        ktomtype: AtomType | None = None,
        ltomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ) -> ImproperType:
        it = ImproperType(
            name, itomtype, jtomtype, ktomtype, ltomtype, *order_params, **kw_params
        )
        self.types[name] = it
        return it


class PairType(Type):

    def __init__(
        self,
        name: str,
        itomtype: int | None,
        jtomtype: int | None,
        *order_params,
        **kw_params,
    ):
        super().__init__(name, *order_params, **kw_params)
        self.itomtype = itomtype
        self.jtomtype = jtomtype

    @property
    def atomtypes(self):
        return [self.itomtype, self.jtomtype]


class PairStyle(Style):

    def def_type(
        self,
        name: str,
        itomtype: AtomType | None = None,
        jtomtype: AtomType | None = None,
        *order_params,
        **kw_params,
    ):
        pt = PairType(name, itomtype, jtomtype, *order_params, **kw_params)
        self.types[name] = pt
        return pt


class ForceField:

    def __init__(self, name: str = ""):

        self.name = name

        self.unit = None
        self.atomstyles: list[AtomStyle] = []
        self.bondstyles: list[BondStyle] = []
        self.pairstyles: list[PairStyle] = []
        self.anglestyles: list[AngleStyle] = []
        self.dihedralstyles: list[DihedralStyle] = []
        self.improperstyles: list[ImproperStyle] = []

    @classmethod
    def from_forcefields(
        cls, *forcefields: "ForceField", name: str = ""
    ) -> "ForceField":
        forcefield = cls(name)
        for ff in forcefields:
            forcefield.merge_(ff)

        return forcefield

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

    def def_atomstyle(self, style: str, *order_params, **kw_params):
        atomstyle = self.get_atomstyle(style)
        if atomstyle:
            return atomstyle
        else:
            atomstyle = AtomStyle(style, *order_params, **kw_params)
            self.atomstyles.append(atomstyle)
        return atomstyle

    def def_bondstyle(self, style: str, *order_params, **kw_params):
        bondstyle = self.get_bondstyle(style)
        if bondstyle:
            return bondstyle
        else:
            bondstyle = BondStyle(style, *order_params, **kw_params)
            self.bondstyles.append(bondstyle)
        return bondstyle

    def def_pairstyle(self, style: str, *order_params, **kw_params):
        pairstyle = self.get_pairstyle(style)
        if pairstyle:
            return pairstyle
        else:
            pairstyle = PairStyle(style, *order_params, **kw_params)
            self.pairstyles.append(pairstyle)
        return pairstyle

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

    def def_anglestyle(self, style: str, *order_params, **kw_params):
        anglestyle = self.get_anglestyle(style)
        if anglestyle:
            return anglestyle
        else:
            anglestyle = AngleStyle(style, *order_params, **kw_params)
            self.anglestyles.append(anglestyle)
        return anglestyle

    def get_diheralstyle(self, style: str):
        for dihedralstyle in self.dihedralstyles:
            if dihedralstyle.name == style:
                return dihedralstyle
        return None

    def def_dihedralstyle(
        self, style: str, *order_params, **kw_params
    ):
        dihe = self.get_diheralstyle(style)
        if dihe:
            return dihe
        else:
            dihe = DihedralStyle(style, *order_params, **kw_params)
            self.dihedralstyles.append(dihe)
        return dihe

    def get_improperstyle(self, style: str):
        for improperstyle in self.improperstyles:
            if improperstyle.name == style:
                return improperstyle
        return None

    def def_improperstyle(
        self, style: str, *order_params, **kw_params
    ):
        improper = self.get_improperstyle(style)
        if improper:
            return improper
        else:
            improper = ImproperStyle(style, *order_params, **kw_params)
            self.improperstyles.append(improper)
        return improper

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
        return reduce(lambda x, y: x + list(y.types.values()), self.atomstyles, [])

    def get_bondtypes(self):
        return reduce(lambda x, y: x + list(y.types.values()), self.bondstyles, [])

    def get_angletypes(self):
        return reduce(lambda x, y: x + list(y.types.values()), self.anglestyles, [])

    def get_dihedraltypes(self):
        return reduce(lambda x, y: x + list(y.types.values()), self.dihedralstyles, [])

    def get_impropertypes(self):
        return reduce(lambda x, y: x + list(y.types.values()), self.improperstyles, [])

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

        ff = ForceField.from_forcefields(self, other)

        return ff

    def merge_(self, other: "ForceField") -> "ForceField":

        def _merge(this_styles, other_styles):
            for style in other_styles:
                matches = [s for s in this_styles if s == style]
                if len(matches) == 1:
                    matches[0].merge(style)
                elif len(matches) == 0:
                    this_styles.append(style)

        _merge(self.atomstyles, other.atomstyles)
        _merge(self.bondstyles, other.bondstyles)
        _merge(self.pairstyles, other.pairstyles)
        _merge(self.anglestyles, other.anglestyles)
        _merge(self.dihedralstyles, other.dihedralstyles)
        _merge(self.improperstyles, other.improperstyles)

        return self

    def __iadd__(self, forcefield: "ForceField"):
        self.merge(forcefield)
        return self

    def __add__(self, forcefield: "ForceField"):
        new_forcefield = ForceField()
        new_forcefield.merge(self)
        new_forcefield.merge(forcefield)
        return new_forcefield

    def get_bond_style(self, name: str):
        for bondstyle in self.bondstyles:
            if bondstyle.name == name:
                return bondstyle

    def get_bond_potential(self, name: str):

        bondstyle = self.get_bond_style(name)
        bond_potential = mp.potential.get_bond_potential(bondstyle)

        return bond_potential
