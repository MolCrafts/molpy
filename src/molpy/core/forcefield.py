from collections import UserDict, defaultdict
from typing import Callable, Union
from functools import reduce
from molpy.core.atomistic import Angle, Atom, Bond, Entity

class DictWithList(UserDict):

    def __init__(self, parms, kwparms):
        super().__init__(kwparms)
        self.parms = parms

    def __getitem__(self, key: int|str):
        """
        Retrieve an item from the dictionary.

        Args:
            key (int|str): The key to retrieve.

        Returns:
            object: The value associated with the key.
        """
        if isinstance(key, int):
            return self.parms[key]
        return super().__getitem__(key)


class Type(DictWithList):

    def __init__(self, name: str, parms: list=[], **kwparms):
        """
        Initialize a ForceField object.

        Args:
            parms: Positional parameters (unnamed, order-dependent).
            kwparms: Keyword parameters (named, as a dict).
        """
        super().__init__(parms, kwparms)
        self._name = name

    def __hash__(self):
        """
        Compute a hash value for the object.

        Returns:
            int: A hash value based on the `name` attribute.
        """
        return hash(self.name)

    def __repr__(self) -> str:
        """
        Provide a string representation of the object.

        Returns:
            str: A string in the format "<ClassName: name>", where `ClassName`
            is the name of the class and `name` is the value of the `name` attribute.
        """
        return f"<{self.__class__.__name__}: {self.name}>"
    
    def __str__(self) -> str:
        return self.name
    
    @property
    def name(self) -> str:
        return self._name
    
    @name.setter
    def name(self, value: str):
        """
        Set the name of the type.

        Args:
            value (str): The new name for the type.
        """
        self._name = value

    def __eq__(self, other: object) -> bool:
        """
        Compare this object with another for equality.

        Args:
            other: The object to compare with.

        Returns:
            bool: True if the `name` attribute of both objects is equal, False otherwise.
        """
        if isinstance(other, str):
            return self.name == other
        if isinstance(other, Type):
            return self.name == other.name
        return False

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

    def get(self, name: str, default=None) -> Type | None:
        """
        Retrieve a type by its name.

        Args:
            name (str): The name of the type to retrieve.
            default: The value to return if the type is not found. Defaults to None.

        Returns:
            Type | None: The type associated with the given name, or the `default` value if not found.
        """
        return next((t for t in self._types if t.name == name), default)

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
        _types = {t.name: t for t in self._types} | {
            t.name: t for t in other._types}
        self._types = list(_types.values())

    def __len__(self):
        """
        Get the number of types in the container.

        Returns:
            int: The number of types in the container.
        """
        return len(self._types)


class Style(DictWithList):

    def __init__(self, name: str, parms=[], **kwparms):
        """
        Initialize a ForceField instance.

        Args:
            name (str): The name of the force field.
            parms: Positional parameters for global configuration.
            kwparms: Keyword parameters for global configuration.

        Attributes:
            name (str): The name of the force field.
            types (TypeContainer): A collection of types.
            parms (tuple): Positional parameters passed during initialization.
        """
        super().__init__(parms, kwparms)
        self.name = name
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

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Style) and self.name == other.name

    def __hash__(self) -> int:
        """
        Compute a hash value for the object.

        Returns:
            int: A hash value based on the `name` attribute.
        """
        return hash(self.name)

    def get_types(self):
        return list(self.types._types)

    def get_by(self, condition: Callable[[Type], bool], default=None) -> Type | None:
        """
        Retrieve the first element from `self.types` that satisfies the given condition.

        Args:
            condition (Callable[[Type], bool]): A function that takes an element and returns a boolean,
                                  used to determine if the element meets the desired criteria.
            default (Optional[Any]): The value to return if no element satisfies the condition.
                                     Defaults to None.

        Returns:
            Type | None: The first element that satisfies the condition, or the `default` value if none is found.
        """
        return next((t for t in self.types if condition(t)), default)

    def get(self, name: str, default=None) -> Type | None:
        """
        Retrieve the type associated with the given name from the forcefield.

        Args:
            name (str): The name of the type to retrieve.
            default (optional): The value to return if the name is not found. Defaults to None.

        Returns:
            Type | None: The type associated with the given name, or the default value if the name is not found.
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
        self.update(other)  # data
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

    def get(self, name: str, default=None) -> Style | None:
        """
        Retrieve a style by its name.

        Args:
            name (str): The name of the style to retrieve.
            default: The value to return if the style is not found. Defaults to None.

        Returns:
            Style | None: The style associated with the given name, or the `default` value if not found.
        """
        return next((s for s in self._styles if s.name == name), default)


class AtomType(Type):
    """
    Represents an atom type in a molecular force field.

    This class is used to define the properties and behavior of a specific type of atom
    within the context of a molecular simulation. It inherits from the `Type` class
    and may include additional attributes or methods specific to atom types.
    """

    def __init__(self, name: str, parms=[], **kwparms):
        """
        Initialize an atom type object.

        Args:
            name (str): The name of the atom type.
            parms: Additional positional parameters.
            kwparms: Additional keyword parameters.
        """
        super().__init__(name, parms, **kwparms)

    def match(self, other: Entity) -> bool:
        """
        Check if the current atom type matches the given atom.

        Args:
            other (Entity): The entity to compare with (should be an Atom).

        Returns:
            bool: True if the types match, False otherwise.
        """
        if hasattr(other, '__getitem__') and 'type' in other:
            return self.name == other["type"]
        return False

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
        itype: AtomType,
        jtype: AtomType,
        name: str = "",
        parms: list = [],
        **kwparms,
    ):
        """
        Initialize a bond type object.

        Args:
            name (str): The name of the forcefield.
            itype (AtomType | None, optional): The atom type for the "i" atom. Defaults to None.
            jtype (AtomType | None, optional): The atom type for the "j" atom. Defaults to None.
            parms: Additional positional parameters.
            kwparms: Additional keyword parameters.
        """
        self.itype = itype
        self.jtype = jtype
        name = name or f"{itype.name}-{jtype.name}"
        super().__init__(name, parms, **kwparms)

    @property
    def atomtypes(self):
        """
        Retrieve the atom types associated with the object.

        Returns:
            list: A list containing the atom types `itype` and `jtype`.
        """
        return [self.itype, self.jtype]

    def match(self, other: Entity) -> bool:
        """
        Check if the current bond type matches the given bond.

        Args:
            other (Entity): The entity to compare with (should be a Bond).

        Returns:
            bool: True if the types match, False otherwise.
        """
        try:
            if hasattr(other, 'itom') and hasattr(other, 'jtom'):
                itom = getattr(other, 'itom')
                jtom = getattr(other, 'jtom')
                if hasattr(itom, '__getitem__') and hasattr(jtom, '__getitem__'):
                    if 'type' in itom and 'type' in jtom:
                        return {itom["type"], jtom["type"]} == {self.itype.name, self.jtype.name}
        except (AttributeError, KeyError, TypeError):
            pass
        return False


class AngleType(Type):

    def __init__(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        name: str = "",
        parms=[],
        **kwparms,
    ):
        """
        Initialize an angle type object.

        Args:
            name (str): The name of the force field.
            itype (AtomType | None, optional): The atom type for the "i" atom. Defaults to None.
            jtype (AtomType | None, optional): The atom type for the "j" atom. Defaults to None.
            ktype (AtomType | None, optional): The atom type for the "k" atom. Defaults to None.
            parms: Additional positional parameters for the base class initialization.
            kwparms: Additional keyword parameters for the base class initialization.
        """
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype
        name = name or f"{itype.name}-{jtype.name}-{ktype.name}"
        super().__init__(
            name, parms, **kwparms
        )

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype]

    def match(self, other: Entity) -> bool:
        """
        Check if the current angle type matches the given angle.

        Args:
            other (Entity): The entity to compare with (should be an Angle).

        Returns:
            bool: True if the types match, False otherwise.
        """
        try:
            if hasattr(other, 'jtom') and hasattr(other, 'itom') and hasattr(other, 'ktom'):
                jtom = getattr(other, 'jtom')
                itom = getattr(other, 'itom')
                ktom = getattr(other, 'ktom')
                if all(hasattr(atom, '__getitem__') for atom in [jtom, itom, ktom]):
                    if all('type' in atom for atom in [jtom, itom, ktom]):
                        return (jtom["type"] == self.jtype.name and 
                               {itom["type"], ktom["type"]} == {self.itype.name, self.ktype.name})
        except (AttributeError, KeyError, TypeError):
            pass
        return False


class DihedralType(Type):

    def __init__(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        ltype: AtomType,
        name: str = "",
        parms=[],
        **kwparms,
    ):
        super().__init__(
            name or "-".join([itype.name, jtype.name, ktype.name, ltype.name]),
            parms,
            **kwparms,
        )
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype
        self.ltype = ltype

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype, self.ltype]

    def match(self, other: Entity) -> bool:
        """
        Check if the current dihedral type matches the given dihedral.

        Args:
            other (Entity): The entity to compare with (should be a Dihedral).

        Returns:
            bool: True if the types match, False otherwise.
        """
        try:
            if all(hasattr(other, attr) for attr in ['itom', 'jtom', 'ktom', 'ltom']):
                atoms = [getattr(other, attr) for attr in ['itom', 'jtom', 'ktom', 'ltom']]
                if all(hasattr(atom, '__getitem__') and 'type' in atom for atom in atoms):
                    atom_types = [atom["type"] for atom in atoms]
                    self_types = [self.itype.name, self.jtype.name, self.ktype.name, self.ltype.name]
                    return atom_types == self_types or atom_types == self_types[::-1]
        except (AttributeError, KeyError, TypeError):
            pass
        return False


class ImproperType(Type):

    def __init__(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        ltype: AtomType,
        name: str = "",
        parms=[],
        **kwparms,
    ):
        self.itype = itype
        self.jtype = jtype
        self.ktype = ktype
        self.ltype = ltype
        name = name or "-".join([a.name for a in self.atomtypes])
        super().__init__(name, parms, **kwparms)

    @property
    def atomtypes(self):
        return [self.itype, self.jtype, self.ktype, self.ltype]

    def match(self, other: Entity) -> bool:
        """
        Check if the current improper type matches the given improper.

        Args:
            other (Entity): The entity to compare with (should be an Improper).

        Returns:
            bool: True if the types match, False otherwise.
        """
        try:
            if all(hasattr(other, attr) for attr in ['itom', 'jtom', 'ktom', 'ltom']):
                atoms = [getattr(other, attr) for attr in ['itom', 'jtom', 'ktom', 'ltom']]
                if all(hasattr(atom, '__getitem__') and 'type' in atom for atom in atoms):
                    atom_types = [atom["type"] for atom in atoms]
                    self_types = [self.itype.name, self.jtype.name, self.ktype.name, self.ltype.name]
                    # For impropers, order might be less strict, so check permutations
                    return atom_types == self_types
        except (AttributeError, KeyError, TypeError):
            pass
        return False


class PairType(Type):

    def __init__(
        self,
        itype: AtomType,
        jtype: AtomType,
        name: str = "",
        parms=[],
        **kwparms,
    ):
        super().__init__(
            name or f"{itype.name}-{jtype.name}", parms=parms, **kwparms
        )
        self.itype = itype
        self.jtype = jtype

    @property
    def atomtypes(self):
        return [self.itype, self.jtype]

    def match(self, other: Entity) -> bool:
        """
        Check if the current pair type matches the given pair interaction.

        Args:
            other (Entity): The entity to compare with (should be a pair interaction).

        Returns:
            bool: True if the types match, False otherwise.
        """
        try:
            if hasattr(other, 'itom') and hasattr(other, 'jtom'):
                itom = getattr(other, 'itom')
                jtom = getattr(other, 'jtom')
                if hasattr(itom, '__getitem__') and hasattr(jtom, '__getitem__'):
                    if 'type' in itom and 'type' in jtom:
                        return {itom["type"], jtom["type"]} == {self.itype.name, self.jtype.name}
        except (AttributeError, KeyError, TypeError):
            pass
        return False

class AtomStyle(Style):

    def __init__(self, name: str, parms, **kwparms):
        super().__init__(name, parms, **kwparms)
        self.classes = defaultdict(set)

    def def_type(self, name: str, class_=None, parms=[], **kwparms) -> AtomType:
        at = AtomType(name, parms, **kwparms)
        self.types.add(at)
        if class_:
            self.classes[class_].add(name)
        return at

    def get_class(self, class_name: str) -> list[AtomType]:
        """
        Retrieve a list of types that belong to a specified class.

        Args:
            class_name (str): The name of the class to filter types by.

        Returns:
            list[Type]: A list of types where the "class" attribute matches the given class_name.
        """
        return list(self.classes.get(class_name, []))


class BondStyle(Style):

    def def_type(
        self,
        itype: AtomType,
        jtype: AtomType,
        name="",
        parms=[],
        **kwparms,
    ) -> BondType:
        bt = BondType(itype, jtype, name=name, parms=parms, **kwparms)
        self.types.add(bt)
        return bt


class AngleStyle(Style):

    def def_type(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        name="",
        parms=[],
        **kwparms,
    ) -> AngleType:
        at = AngleType(itype, jtype, ktype, name=name, parms=parms, **kwparms)
        self.types.add(at)
        return at


class DihedralStyle(Style):

    def def_type(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        ltype: AtomType,
        name="",
        parms=[],
        **kwparms,
    ) -> DihedralType:
        dt = DihedralType(itype, jtype, ktype, ltype, name=name, parms=parms, **kwparms)
        self.types.add(dt)
        return dt


class ImproperStyle(Style):

    def def_type(
        self,
        itype: AtomType,
        jtype: AtomType,
        ktype: AtomType,
        ltype: AtomType,
        name="",
        parms=[],
        **kwparms,
    ) -> ImproperType:
        it = ImproperType(itype, jtype, ktype, ltype, name=name, parms=parms, **kwparms)
        self.types.add(it)
        return it

class PairStyle(Style):

    def def_type(
        self,
        itype: AtomType,
        jtype: AtomType,
        name="",
        parms=[],
        **kwparms,
    ):
        pt = PairType(itype, jtype, name=name, parms=parms, **kwparms)
        self.types.add(pt)
        return pt


class ForceField:
    """
    ForceField class represents a molecular force field, which defines the styles and types
    of interactions between atoms, bonds, angles, dihedrals, and impropers in a molecular system.
    """

    def __init__(self, name: str = "", unit: str = "real"):

        self.name = name
        self.unit = unit
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
        if len(self.atomstyles) > 0:
            detail += (
                f"\nn_atomstyles: {len(self.atomstyles)}, n_atomtypes: {len(self.get_atomtypes())}"
            )
        if len(self.bondstyles) > 0:
            detail += (
                f"\nn_bondstyles: {len(self.bondstyles)}, n_bondtypes: {len(self.get_bondtypes())}"
            )
        if len(self.pairstyles) > 0:
            detail += (
                f"\nn_pairstyles: {len(self.pairstyles)}, n_pairtypes: {len(self.get_pairtypes())}"
            )
        if len(self.anglestyles) > 0:
            detail += f"\nn_anglestyles: {len(self.anglestyles)}, n_angletypes: {len(self.get_angletypes())}"
        if len(self.dihedralstyles) > 0:
            detail += f"\nn_dihedralstyles: {len(self.dihedralstyles)}, n_dihedraltypes: {len(self.get_dihedraltypes())}"
        if len(self.improperstyles) > 0:
            detail += f"\nn_improperstyles: {len(self.improperstyles)}, n_impropertypes: {len(self.get_impropertypes())}"
        return detail + ">"

    @property
    def n_atomstyles(self) -> int:
        """
        Get the number of atom styles in the force field.

        Returns:
            int: The number of atom styles.
        """
        return len(self.atomstyles)
    
    @property
    def n_bondstyles(self) -> int:
        """
        Get the number of bond styles in the force field.

        Returns:
            int: The number of bond styles.
        """
        return len(self.bondstyles)

    @property
    def n_pairstyles(self) -> int:
        """
        Get the number of pair styles in the force field.

        Returns:
            int: The number of pair styles.
        """
        return len(self.pairstyles)
    
    @property
    def n_anglestyles(self) -> int:
        """
        Get the number of angle styles in the force field.

        Returns:
            int: The number of angle styles.
        """
        return len(self.anglestyles)

    @property
    def n_dihedralstyles(self) -> int:
        """
        Get the number of dihedral styles in the force field.

        Returns:
            int: The number of dihedral styles.
        """
        return len(self.dihedralstyles)

    @property
    def n_improperstyles(self) -> int:
        """
        Get the number of improper styles in the force field.

        Returns:
            int: The number of improper styles.
        """
        return len(self.improperstyles)

    @property
    def n_atomtypes(self) -> int:
        """
        Get the number of atom types in the force field.

        Returns:
            int: The number of atom types.
        """
        return len(self.get_atomtypes())

    @property
    def n_bondtypes(self) -> int:
        """
        Get the number of bond types in the force field.

        Returns:
            int: The number of bond types.
        """
        return len(self.get_bondtypes())

    @property
    def n_angletypes(self) -> int:
        """
        Get the number of angle types in the force field.

        Returns:
            int: The number of angle types.
        """
        return len(self.get_angletypes())

    @property
    def n_dihedraltypes(self) -> int:
        """
        Get the number of dihedral types in the force field.

        Returns:
            int: The number of dihedral types.
        """
        return len(self.get_dihedraltypes())

    @property
    def n_impropertypes(self) -> int:
        """
        Get the number of improper types in the force field.

        Returns:
            int: The number of improper types.
        """
        return len(self.get_impropertypes())

    @property
    def n_pairtypes(self) -> int:
        """
        Get the number of pair types in the force field.

        Returns:
            int: The number of pair types.
        """
        return len(self.get_pairstypes())

    def def_atomstyle(self, name: str, parms=[], **data):
        atomstyle = self.get_atomstyle(name)
        if atomstyle:
            return atomstyle
        else:
            atomstyle = AtomStyle(name, parms, **data)
            self.atomstyles.append(atomstyle)
        return atomstyle

    def def_bondstyle(self, style: str, parms=[], **data):
        bondstyle = self.get_bondstyle(style)
        if bondstyle is not None:
            return bondstyle
        else:
            bondstyle = BondStyle(style, parms, **data)
            self.bondstyles.append(bondstyle)
        return bondstyle

    def def_anglestyle(self, style: str, parms=[], **data) -> AngleStyle:
        """
        Define or retrieve an angle style by its name.

        Args:
            style (str): The name of the angle style.
            *parms: Positional parameters for the angle style.
            **data: Keyword parameters for the angle style.

        Returns:
            AngleStyle: The defined or retrieved AngleStyle object.
        """
        anglestyle = self.get_anglestyle(style)
        if anglestyle is not None:
            return anglestyle
        else:
            anglestyle = AngleStyle(style, parms, **data)
            self.anglestyles.append(anglestyle)
        return anglestyle

    def def_dihedralstyle(self, style: str, parms=[], **data):
        dihe = self.get_dihedralstyle(style)
        if dihe is not None:
            return dihe
        else:
            dihe = DihedralStyle(style, parms, **data)
            self.dihedralstyles.append(dihe)
        return dihe

    def def_improperstyle(self, style: str, parms=[], **data):
        improper = self.get_improperstyle(style)
        if improper is not None:
            return improper
        else:
            improper = ImproperStyle(style, parms, **data)
            self.improperstyles.append(improper)
        return improper

    def def_pairstyle(self, style: str, parms=[], **data):
        pairstyle = self.get_pairstyle(style)
        if pairstyle is not None:
            return pairstyle
        else:
            pairstyle = PairStyle(style, parms, **data)
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
            (improperstyle for improperstyle in self.improperstyles if improperstyle.name == name),
            None,
        )

    def get_pairstyle(self, name: str) -> PairStyle | None:
        """
        Retrieve a pair style by its name.

        Args:
            name (str): The name of the pair style to retrieve.

        Returns:
            PairStyle | None: The PairStyle object if found, otherwise None.
        """
        return next(
            (pairstyle for pairstyle in self.pairstyles if pairstyle.name == name),
            None,
        )

    def get_atomtypes(self) -> list[AtomType]:
        """
        Get all atom types from all atomstyles.
        Returns:
            list[AtomType]: All atom types in the forcefield.
        """
        return reduce(
            lambda x, y: x + y,
            [atomstyle.get_types() for atomstyle in self.atomstyles],
            [],
        )

    def get_bondtypes(self) -> list[BondType]:
        """
        Get all bond types from all bondstyles.
        Returns:
            list[BondType]: All bond types in the forcefield.
        """
        bondtypes = []
        for bondstyle in self.bondstyles:
            bondtypes.extend(bondstyle.get_types())
        return bondtypes

    def get_angletypes(self) -> list[AngleType]:
        """
        Get all angle types from all anglestyles.
        Returns:
            list[AngleType]: All angle types in the forcefield.
        """
        angletypes = []
        for angstyle in self.anglestyles:
            angletypes.extend(angstyle.get_types())
        return angletypes

    def get_dihedraltypes(self) -> list[DihedralType]:
        """
        Get all dihedral types from all dihedralstyles.
        Returns:
            list[DihedralType]: All dihedral types in the forcefield.
        """
        dihedraltypes = []
        for dihe in self.dihedralstyles:
            dihedraltypes.extend(dihe.get_types())
        return dihedraltypes

    def get_impropertypes(self) -> list[ImproperType]:
        """
        Get all improper types from all improperstyles.
        Returns:
            list[ImproperType]: All improper types in the forcefield.
        """
        impropertypes = []
        for imp in self.improperstyles:
            impropertypes.extend(imp.get_types())
        return impropertypes

    def get_pairtypes(self) -> list[PairType]:
        """
        Get all pair types from all pairstyles.
        Returns:
            list[PairType]: All pair types in the forcefield.
        """
        pairtypes = []
        for pairstyle in self.pairstyles:
            pairtypes.extend(pairstyle.get_types())
        return pairtypes

    def __contains__(self, name: str) -> bool:
        """Check if a style or type exists by name."""
        return (
            self.get_atomstyle(name) is not None or
            self.get_bondstyle(name) is not None or
            self.get_anglestyle(name) is not None or
            self.get_dihedralstyle(name) is not None or
            self.get_improperstyle(name) is not None or
            self.get_pairstyle(name) is not None
        )

    def __getitem__(self, name: str) -> Style:
        """Get a style by name (atom, bond, angle, dihedral, improper, pair)."""
        for getter in [self.get_atomstyle, self.get_bondstyle, self.get_anglestyle, self.get_dihedralstyle, self.get_improperstyle, self.get_pairstyle]:
            style = getter(name)
            if style is not None:
                return style
        raise KeyError(f"No style named '{name}' found.")

    def __len__(self) -> int:
        """Return the number of styles in the forcefield."""
        return sum([
            len(self.atomstyles),
            len(self.bondstyles),
            len(self.anglestyles),
            len(self.dihedralstyles),
            len(self.improperstyles),
            len(self.pairstyles),
        ])

    def merge(self, other: "ForceField"):
        """
        Merge another ForceField into this one (in-place).
        Args:
            other (ForceField): The other forcefield to merge.
        Returns:
            self
        """
        for style in other.atomstyles:
            if self.get_atomstyle(style.name) is None:
                self.atomstyles.append(style)
        for style in other.bondstyles:
            if self.get_bondstyle(style.name) is None:
                self.bondstyles.append(style)
        for style in other.anglestyles:
            if self.get_anglestyle(style.name) is None:
                self.anglestyles.append(style)
        for style in other.dihedralstyles:
            if self.get_dihedralstyle(style.name) is None:
                self.dihedralstyles.append(style)
        for style in other.improperstyles:
            if self.get_improperstyle(style.name) is None:
                self.improperstyles.append(style)
        for style in other.pairstyles:
            if self.get_pairstyle(style.name) is None:
                self.pairstyles.append(style)
        return self

    def merge_(self, other: "ForceField"):
        """
        Merge another ForceField into this one (in-place, alias for merge).
        """
        return self.merge(other)
