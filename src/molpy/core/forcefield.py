from collections import defaultdict
from copy import deepcopy
from functools import reduce
from typing import Callable, Union

from molpy.core.struct import Angle, Atom, Bond, Entity


class Type:

    def __init__(self, label, oparam: list, **param):
        """
        Initialize a ForceField object.

        Args:
            name (str): The name of the force field.
            oparam: Positional parameters without keys.
            param: Keyword parameters with keys.
        """
        self.label = label
        self.param = param
        self.oparam = oparam

    def __hash__(self):
        """
        Compute a hash value for the object.

        Returns:
            int: A hash value based on the `name` attribute.
        """
        return hash(self.label)

    def __repr__(self) -> str:
        """
        Provide a string representation of the object.

        Returns:
            str: A string in the format "<ClassName: name>", where `ClassName`
            is the name of the class and `name` is the value of the `name` attribute.
        """
        return f"<{self.__class__.__name__}: {self.label}>"

    def __eq__(self, other: Union["Type", str]):
        """
        Compare this object with another for equality.

        Args:
            other (Type): The object to compare with.

        Returns:
            bool: True if the `name` attribute of both objects is equal, False otherwise.
        """
        if isinstance(other, str):
            return self.label == other
        return self.label == other.label
    
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.param.get(key)
        elif isinstance(key, int):
            return self.oparam[key]
        else:
            raise TypeError(f"Invalid key type: {type(key)}. Expected str or int.")

    def match(self, other: Entity) -> bool:
        """
        Check if the current type matches the given entity.

        Args:
            other (Entity): The entity to compare with.

        Returns:
            bool: True if the types match, False otherwise.
        """
        raise NotImplementedError(
            "The match method should be implemented in subclasses of Type."
        )

    def apply(self, other: Entity):
        """
        Apply the type to the given entity.

        Args:
            other (Entity): The entity to apply the type to.
        """
        raise NotImplementedError(
            "The apply method should be implemented in subclasses of Type."
        )


class TypeContainer:

    def __init__(self):
        self._types = list()

    def add(self, t: Type):
        """
        Add a type to the container.

        Args:
            t (Type): The type to be added.
        """
        self._types.append(t)

    def __iter__(self):
        """
        Iterate over the types in the container.

        Returns:
            iterator: An iterator over the types in the container.
        """
        return iter(self._types)

    def get(self, label: str, default=None) -> Type:
        """
        Retrieve a type by its label.

        Args:
            label (str): The label of the type to retrieve.
            default: The value to return if the type is not found. Defaults to None.

        Returns:
            Type: The type associated with the given label, or the `default` value if not found.
        """
        return next((t for t in self._types if t.label == label), default)

    def get_all_by(self, condition: Callable) -> list[Type]:
        """
        Retrieve all types that satisfy a given condition.

        Args:
            condition (Callable): A function that takes a type as input and returns
                                  a boolean indicating whether the type satisfies the condition.

        Returns:
            list[Type]: A list of types that meet the specified condition.
        """
        return [t for t in self._types if condition(t)]

    def update(self, other):
        """
        Update the container with types from another container.

        Args:
            other (TypeContainer): Another TypeContainer object to merge with.
        """
        self._types.update(other._types)

    def __len__(self):
        """
        Get the number of types in the container.

        Returns:
            int: The number of types in the container.
        """
        return len(self._types)


class Style:

    def __init__(self, name: str, oparam=[], **param):
        """
        Initialize a ForceField instance.

        Args:
            name (str): The name of the force field.
            *oparam: Positional parameters for global configuration.
            **param: Keyword parameters for global configuration.

        Attributes:
            name (str): The name of the force field.
            types (TypeContainer): A collection of types.
            oparam (tuple): Positional parameters passed during initialization.
        """
        self.name = name
        self.oparam = oparam
        self.param = param
        self.types: TypeContainer = TypeContainer()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    @property
    def n_types(self):
        """
        Calculate the number of unique types in the forcefield.

        Returns:
            int: The number of unique types.
        """
        return len(self.types)

    def __eq__(self, other: "Style"):
        return self.name == other.name
    
    def get_types(self):
        return list(self.types._types)

    def get_by(self, condition: Callable[[Type], bool], default=None) -> Type:
        """
        Retrieve the first element from `self.types` that satisfies the given condition.

        Args:
            condition (Callable[[Type], bool]): A function that takes an element and returns a boolean,
                                  used to determine if the element meets the desired criteria.
            default (Optional[Any]): The value to return if no element satisfies the condition.
                                     Defaults to None.

        Returns:
            Type: The first element that satisfies the condition, or the `default` value if none is found.
        """
        return next((t for t in self.types if condition(t)), default)

    def get(self, name: str, default=None) -> Type:
        """
        Retrieve the type associated with the given name from the forcefield.

        Args:
            name (str): The name of the type to retrieve.
            default (optional): The value to return if the name is not found. Defaults to None.

        Returns:
            Type: The type associated with the given name, or the default value if the name is not found.
        """
        return self.types.get(name, default)

    def get_all_by(self, condition: Callable) -> list[Type]:
        """
        Retrieve all items from the `types` dictionary that satisfy a given condition.

        Args:
            condition (Callable): A function that takes an item as input and returns
                                  a boolean indicating whether the item satisfies the condition.

        Returns:
            list[Type]: A list of items from the `types` dictionary that meet the specified condition.
        """
        return self.types.get_all_by(condition)

    def merge(self, other: "Style"):
        """
        Merges the current Style object with another Style object.

        This method updates the current Style object with the attributes and types
        from the provided `other` Style object. If there are overlapping type names,
        the types from the `other` Style object will overwrite those in the current
        object.

        Args:
            other (Style): Another Style object to merge into the current one.

        Returns:
            Style: The updated Style object after merging.
        """
        self.update(other)  # param
        self.types.update(other.types)
        return self


class StyleContainer:

    def __init__(self):
        self._styles = set()

    def add(self, style: Style):
        """
        Add a style to the container.

        Args:
            style (Style): The style to be added.
        """
        self._styles.add(style)

    def __iter__(self):
        """
        Iterate over the styles in the container.

        Returns:
            iterator: An iterator over the styles in the container.
        """
        return iter(self._styles)

    def get(self, name: str, default=None) -> Style:
        """
        Retrieve a style by its name.

        Args:
            name (str): The name of the style to retrieve.
            default: The value to return if the style is not found. Defaults to None.

        Returns:
            Style: The style associated with the given name, or the `default` value if not found.
        """
        return next((s for s in self._styles if s.name == name), default)


class AtomType(Type):
    """
    Represents an atom type in a molecular force field.

    This class is used to define the properties and behavior of a specific type of atom
    within the context of a molecular simulation. It inherits from the `Type` class
    and may include additional attributes or methods specific to atom types.
    """

    def __init__(self, label: str, oparam=[], **param):
        """
        Initialize an atom type object.

        Args:
            label (str): The label of the atom type.
            oparam: Additional positional parameters.
            param: Additional keyword parameters.
        """
        super().__init__(label, oparam=oparam, **param)

    def match(self, other: Atom) -> bool:
        """
        Check if the current atom type matches the given atom.

        Args:
            other (Atom): The atom to compare with.

        Returns:
            bool: True if the types match, False otherwise.
        """
        return self.label == other.type

    def apply(self, other: Atom):
        """
        Apply the atom type to the given atom.

        Args:
            other (Atom): The atom to apply the type to.
        """
        other.update(self)


class BondType(Type):

    def __init__(
        self,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        label: str = "",
        oparam: list = [],
        **param,
    ):
        """
        Initialize a bond type object.

        Args:
            name (str): The name of the forcefield.
            itype (AtomType | None, optional): The atom type for the "i" atom. Defaults to None.
            jtype (AtomType | None, optional): The atom type for the "j" atom. Defaults to None.
            oparam: Additional positional parameters.
            param: Additional keyword parameters.
        """
        super().__init__(label, oparam=oparam, **param)
        self.itype = itype
        self.jtype = jtype
        self.label = label or f"{itype.label}-{jtype.label}"

    @property
    def atomtypes(self):
        """
        Retrieve the atom types associated with the object.

        Returns:
            list: A list containing the atom types `itype` and `jtype`.
        """
        return [self.itype, self.jtype]

    def match(self, other: Bond) -> bool:
        """
        Check if the current bond type matches the given bond.

        Args:
            other (Bond): The bond to compare with.

        Returns:
            bool: True if the types match, False otherwise.
        """
        return {other.itom["type"], other.jtom["type"]} == {self.itype, self.jtype}

    def apply(self, other: Bond):
        """
        Apply the bond type to the given bond.

        Args:
            other (Bond): The bond to apply the type to.
        """
        other["type"] = self

class AngleType(Type):

    def __init__(
        self,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        label: str = "",
        oparam=[],
        **param,
    ):
        """
        Initialize an angle type object.

        Args:
            name (str): The name of the force field.
            itype (AtomType | None, optional): The atom type for the "i" atom. Defaults to None.
            jtype (AtomType | None, optional): The atom type for the "j" atom. Defaults to None.
            ktype (AtomType | None, optional): The atom type for the "k" atom. Defaults to None.
            *oparam: Additional positional parameters for the base class initialization.
            **param: Additional keyword parameters for the base class initialization.
        """
        super().__init__(label or f"{itype.label}-{jtype.label}-{ktype.label}", oparam, **param)
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype]

    def match(self, other: Angle) -> bool:
        """
        Check if the current angle type matches the given angle.

        Args:
            other (Angle): The angle to compare with.

        Returns:
            bool: True if the types match, False otherwise.
        """
        return self.jtype == other.jtom.type and set(self.itype, self.ktype) == set(
            other.itom.type, other.ktom.type
        )


class DihedralType(Type):

    def __init__(
        self,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        ltype: AtomType | None = None,
        label: str = "",
        oparam=[],
        **param,
    ):
        """
        Represents a dihedral type in a molecular force field, which defines the
        interaction parameters for a set of four connected atoms.

        Args:
            name (str): The name of the dihedral type.
            itype (AtomType | None): The atom type of the first atom in the dihedral.
            jtype (AtomType | None): The atom type of the second atom in the dihedral.
            ktype (AtomType | None): The atom type of the third atom in the dihedral.
            ltype (AtomType | None): The atom type of the fourth atom in the dihedral.
        """
        super().__init__(label or "-".join([itype.label, jtype.label, ktype.label, ltype.label]), oparam, **param)
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype
        self.ltype = ltype

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype, self.ltype]


class ImproperType(Type):

    def __init__(
        self,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        ltype: AtomType | None = None,
        *oparam,
        **param,
    ):
        super().__init__(oparam, param)
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype
        self.ltype = ltype

    @property
    def label(self) -> str:
        """
        Retrieve the label of the improper type.
        """
        return self.get("label", "-".join([a.label for a in self.atomtypes]))

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype, self.ltype]


class AtomStyle(Style):

    def __init__(self, name: str, oparam, **param):
        super().__init__(name, oparam, **param)
        self.classes = defaultdict(set)

    def def_type(self, label: str, class_=None, oparam=[], **param) -> AtomType:
        at = AtomType(label, oparam, **param)
        self.types.add(at)
        if class_:
            self.classes[class_].add(label)
        return at

    def get_class(self, class_name: str) -> list[AtomType]:
        """
        Retrieve a list of types that belong to a specified class.

        Args:
            class_name (str): The name of the class to filter types by.

        Returns:
            list[Type]: A list of types where the "class" attribute matches the given class_name.
        """
        return self.classes.get(class_name, [])


class BondStyle(Style):

    def def_type(
        self,
        name: str,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        oparam=[],
        **param,
    ) -> BondType:
        bt = BondType(name, itype, jtype, oparam, **param)
        self.types.add(bt)
        return bt


class AngleStyle(Style):

    def def_type(
        self,
        name: str,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        oparam=[],
        **param,
    ) -> AngleType:
        at = AngleType(name, itype, jtype, ktype, oparam, **param)
        self.types.add(at)
        return at


class DihedralStyle(Style):

    def def_type(
        self,
        name: str,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        ltype: AtomType | None = None,
        oparam=[],
        **param,
    ) -> DihedralType:
        dt = DihedralType(name, itype, jtype, ktype, ltype, oparam, **param)
        self.types.add(dt)
        return dt


class ImproperStyle(Style):

    def def_type(
        self,
        name: str,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        ktype: AtomType | None = None,
        ltype: AtomType | None = None,
        oparam=[],
        param={},
    ) -> ImproperType:
        it = ImproperType(name, itype, jtype, ktype, ltype, oparam, param)
        self.types.add(it)
        return it


class PairType(Type):

    def __init__(
        self,
        name: str,
        itype: int | None,
        jtype: int | None,
        oparam=[],
        **param,
    ):
        super().__init__(name, oparam, **param)
        self.itype = itype
        self.jtype = jtype

    @property
    def atomtypes(self):
        return [self.itype, self.jtype]


class PairStyle(Style):

    def def_type(
        self,
        name: str,
        itype: AtomType | None = None,
        jtype: AtomType | None = None,
        oparam=[],
        **param,
    ):
        pt = PairType(name, itype, jtype, oparam, **param)
        self.types.add(pt)
        return pt


class ForceField:
    """
    ForceField class represents a molecular force field, which defines the styles and types
    of interactions between atoms, bonds, angles, dihedrals, and impropers in a molecular system.
    """

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
        cls, name: str = "", *forcefields: "ForceField"
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

    def def_atomstyle(self, name: str, oparam=[], **param):
        atomstyle = self.get_atomstyle(name)
        if atomstyle:
            return atomstyle
        else:
            atomstyle = AtomStyle(name, oparam, **param)
            self.atomstyles.append(atomstyle)
        return atomstyle

    def def_bondstyle(self, style: str, oparam=[], **param):
        bondstyle = self.get_bondstyle(style)
        if bondstyle:
            return bondstyle
        else:
            bondstyle = BondStyle(style, oparam, **param)
            self.bondstyles.append(bondstyle)
        return bondstyle

    def def_anglestyle(self, style: str, oparam=[], **param) -> AngleStyle:
        """
        Define or retrieve an angle style by its name.

        Args:
            style (str): The name of the angle style.
            *oparam: Positional parameters for the angle style.
            **param: Keyword parameters for the angle style.

        Returns:
            AngleStyle: The defined or retrieved AngleStyle object.
        """
        anglestyle = self.get_anglestyle(style)
        if not anglestyle:
            anglestyle = AngleStyle(style, oparam, **param)
            self.anglestyles.append(anglestyle)
        return anglestyle

    def def_dihedralstyle(self, style: str, oparam=[], **param):
        dihe = self.get_dihedralstyle(style)
        if dihe:
            return dihe
        else:
            dihe = DihedralStyle(style, oparam, **param)
            self.dihedralstyles.append(dihe)
        return dihe

    def def_improperstyle(self, style: str, oparam=[], **param):
        improper = self.get_improperstyle(style)
        if improper:
            return improper
        else:
            improper = ImproperStyle(style, oparam, **param)
            self.improperstyles.append(improper)
        return improper

    def def_pairstyle(self, style: str, oparam=[], **param):
        pairstyle = self.get_pairstyle(style)
        if pairstyle:
            return pairstyle
        else:
            pairstyle = PairStyle(style, oparam, **param)
            self.pairstyles.append(pairstyle)
        return pairstyle

    def get_atomstyle(self, name: str) -> AtomStyle | None:
        """
        Retrieve an atom style by its name.

        Args:
            name (str): The name of the atom style to retrieve.

        Returns:
            AtomStyle | None: The AtomStyle object if found, otherwise None.
        """
        return next(
            (atomstyle for atomstyle in self.atomstyles if atomstyle.name == name), None
        )

    def get_bondstyle(self, name: str):
        """
        Retrieve a bond style by its name.

        Args:
            name (str): The name of the bond style to retrieve.

        Returns:
            BondStyle | None: The BondStyle object if found, otherwise None.
        """
        return next(
            (bondstyle for bondstyle in self.bondstyles if bondstyle.name == name), None
        )

    def get_anglestyle(self, name: str) -> AngleStyle | None:
        """
        Retrieve an angle style by its name.

        Args:
            name (str): The name of the angle style to retrieve.

        Returns:
            AngleStyle | None: The AngleStyle object if found, otherwise None.
        """
        return next(
            (anglestyle for anglestyle in self.anglestyles if anglestyle.name == name),
            None,
        )

    def get_dihedralstyle(self, name: str) -> DihedralStyle | None:
        """
        Retrieve a dihedral style by its name.

        Args:
            name (str): The name of the dihedral style to retrieve.

        Returns:
            DihedralStyle | None: The DihedralStyle object if found, otherwise None.
        """
        return next(
            (
                dihedralstyle
                for dihedralstyle in self.dihedralstyles
                if dihedralstyle.name == name
            ),
            None,
        )

    def get_improperstyle(self, name: str) -> ImproperStyle | None:
        """
        Retrieve an improper style by its name.

        Args:
            name (str): The name of the improper style to retrieve.

        Returns:
            ImproperStyle | None: The ImproperStyle object if found, otherwise None.
        """
        return next(
            (
                improperstyle
                for improperstyle in self.improperstyles
                if improperstyle.name == name
            ),
            None,
        )

    def get_pairstyle(self, name: str):
        """
        Retrieve a pair style object by its name.

        Args:
            name (str): The name of the pair style to retrieve.

        Returns:
            The pair style object with the specified name if found,
            otherwise None.
        """
        return next(
            (pairstyle for pairstyle in self.pairstyles if pairstyle.name == name), None
        )

    def get_atomtypes(self):
        """
        Retrieve a list of atom types from the atom styles.

        This method aggregates all atom types from the `types` attribute of each
        atom style in the `atomstyles` collection.

        Returns:
            list: A list containing all atom types extracted from the `types`
                  dictionaries of the atom styles.
        """
        return reduce(lambda x, y: x + list(y.types), self.atomstyles, [])

    def get_bondtypes(self):
        """
        Retrieve all bond types from the bond styles.

        This method aggregates bond types from all bond styles by iterating
        through the `types` attribute of each bond style and collecting their
        values into a single list.

        Returns:
            list: A list containing all bond types from the bond styles.
        """
        return reduce(lambda x, y: x + list(y.types), self.bondstyles, [])

    def get_angletypes(self):
        """
        Retrieve all angle types from the defined angle styles.

        This method iterates through the `anglestyles` attribute, which is expected
        to be a collection of objects containing a `types` dictionary. It extracts
        the values from each `types` dictionary, combines them into a single list,
        and returns the result.

        Returns:
            list: A list of all angle types extracted from the `anglestyles` attribute.
        """
        return reduce(lambda x, y: x + list(y.types), self.anglestyles, [])

    def get_dihedraltypes(self):
        """
        Retrieve a list of dihedral types from the dihedral styles.

        This method aggregates all dihedral types from the `dihedralstyles` attribute,
        which is expected to be an iterable of objects. Each object in `dihedralstyles`
        should have a `types` attribute that is a dictionary. The values of these
        dictionaries are collected and combined into a single list.

        Returns:
            list: A list containing all dihedral types extracted from the `dihedralstyles`.
        """
        return reduce(lambda x, y: x + list(y.types), self.dihedralstyles, [])

    def get_impropertypes(self):
        """
        Retrieve a list of improper types from the improper styles.

        This method aggregates all improper types defined in the `improperstyles`
        attribute by iterating through each improper style and collecting the
        values of their `types` dictionaries.

        Returns:
            list: A list of improper types aggregated from all improper styles.
        """
        return reduce(lambda x, y: x + list(y.types), self.improperstyles, [])

    @property
    def n_atomstyles(self):
        """
        Calculate the number of atom styles.
        """
        return len(self.atomstyles)

    @property
    def n_bondstyles(self):
        """
        Calculate the number of bond styles.
        """
        return len(self.bondstyles)

    @property
    def n_pairstyles(self):
        """
        Calculate the number of pair styles.
        """
        return len(self.pairstyles)

    @property
    def n_anglestyles(self):
        """
        Calculate the number of angle styles.
        """
        return len(self.anglestyles)

    @property
    def n_dihedralstyles(self):
        """
        Calculate the number of dihedral styles.
        """
        return len(self.dihedralstyles)

    @property
    def n_improperstyles(self):
        """
        Calculate the number of improper styles.
        """
        return len(self.improperstyles)

    @property
    def n_atomtypes(self):
        """
        Calculate the number of atom types.
        """
        return reduce(lambda x, y: x + y.n_types, self.atomstyles, 0)

    @property
    def n_bondtypes(self):
        """
        Calculate the number of bond types.
        """
        return reduce(lambda x, y: x + y.n_types, self.bondstyles, 0)

    @property
    def n_angletypes(self):
        """
        Calculate the number of angle types.
        """
        return reduce(lambda x, y: x + y.n_types, self.anglestyles, 0)

    @property
    def n_dihedraltypes(self):
        """
        Calculate the number of dihedral types.
        """
        return reduce(lambda x, y: x + y.n_types, self.dihedralstyles, 0)

    @property
    def n_impropertypes(self):
        """
        Calculate the number of improper types.
        """
        return reduce(lambda x, y: x + y.n_types, self.improperstyles, 0)

    @property
    def n_pairtypes(self):
        """
        Calculate the number of pair types.
        """
        return reduce(lambda x, y: x + y.n_types, self.pairstyles, 0)

    @property
    def atomtypes(self):
        """
        Calculate the list of atom types.
        """
        return reduce(lambda x, y: x + list(y.types), self.atomstyles, list())

    @property
    def bondtypes(self):
        """
        Calculate the list of bond types.
        """
        return reduce(lambda x, y: x + list(y.types), self.bondstyles, list())

    @property
    def angletypes(self):
        """
        Calculate the list of angle types.
        """
        return reduce(lambda x, y: x + list(y.types), self.anglestyles, list())

    @property
    def dihedraltypes(self):
        """
        Calculate the list of dihedral types.
        """
        return reduce(lambda x, y: x + list(y.types), self.dihedralstyles, list())

    @property
    def impropertypes(self):
        """
        Calculate the list of improper types.
        """
        return reduce(lambda x, y: x + list(y.types), self.improperstyles, list())

    @property
    def pairtypes(self):
        """
        Calculate the number of pair types.
        """
        return reduce(lambda x, y: x + list(y.types), self.pairstyles, list())

    def merge(self, other: "ForceField") -> "ForceField":
        """
        Merges the current ForceField instance with another ForceField instance.

        This method combines the styles (e.g., atomstyles, bondstyles, pairstyles,
        anglestyles, dihedralstyles, improperstyles) of the current ForceField
        object with those of another ForceField object. If a style in the other
        ForceField matches an existing style in the current ForceField, the styles
        are merged. Otherwise, the style from the other ForceField is added to the
        current ForceField.

        Args:
            other (ForceField): The other ForceField instance to merge with.

        Returns:
            ForceField: The updated ForceField instance after merging.
        """
        other = deepcopy(other)

        def _merge(this_styles: list[Style], other_styles: list[Style]):
            for style in other_styles:
                matches = [s for s in this_styles if s == style]
                if matches:
                    matches[0].merge(style)
                else:
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
        return self.merge(forcefield)
