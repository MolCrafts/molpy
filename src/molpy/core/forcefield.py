from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Self, TypeVar, List, cast

from .entity import Entity

# --- Generic variables and TypeBucket implementation ---

T = TypeVar("T")
S = TypeVar("S", bound="Style")
Ty = TypeVar("Ty", bound="Type")


def get_nearest_type[T](item: T) -> type[T]:
    return type(item)


class TypeBucket[T]:
    def __init__(self) -> None:
        self._items: dict[type[T], set[T]] = defaultdict(set)

    def add(self, item: T) -> None:
        cls = get_nearest_type(item)
        self._items[cls].add(item)

    def remove(self, item: T) -> None:
        cls = get_nearest_type(item)
        self._items[cls].discard(item)

    def bucket(self, cls: type) -> List[T]:
        result: List[T] = []
        for k, items in self._items.items():
            # runtime check: k is a class, cls is a class
            if issubclass(k, cls):
                result.extend(list(items))
        return result

    def classes(self) -> Iterator[type[T]]:
        return iter(self._items.keys())


# --- Basic components ---


class Parameters:
    def __init__(self, *args: Any, **kwargs: Any):
        self.args = list(args)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"Parameters(args={self.args}, kwargs={self.kwargs})"

    def __getitem__(self, key: int | slice | str):
        if isinstance(key, str):
            return self.kwargs[key]
        else:
            return self.args[key]


class Type:
    def __init__(self, name: str, *args: Any, **kwargs: Any):
        self._name = name
        self.params = Parameters(*args, **kwargs)

    @property
    def name(self) -> str:
        return self._name

    def __hash__(self) -> int:
        return hash((self.__class__, self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Type):
            return False
        return self.__class__ == other.__class__ and self.name == other.name

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Type):
            return False
        return self.__class__ == other.__class__ and self.name > other.name

    def __getitem__(self, key: str) -> Any:
        return self.params.kwargs.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        self.params.kwargs[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.params.kwargs

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.kwargs.get(key, default)


class Style:
    def __init__(self, name: str, *args: Any, **kwargs: Any):
        self.name = name
        self.params = Parameters(*args, **kwargs)
        self.types = TypeBucket[Type]()

    def __hash__(self) -> int:
        return hash((self.__class__, self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self.name == other.name

    def merge(self, other: "Style"):
        self.params.args.extend(other.params.args)
        self.params.kwargs.update(other.params.kwargs)
        for t in other.types.bucket(Type):
            self.types.add(t)
        return self


# ===================================================================
#                    ForceField base class
# ===================================================================


class ForceField:
    # Kernel registry for potential functions
    _kernel_registry: dict[str, dict[str, type]] = {}

    def __init__(self, name: str = "", units: str = "real"):
        self.name = name
        self.units = units
        self.styles = TypeBucket[Style]()

    def def_style(self, style: Style) -> Style:
        """Register a Style instance with the force field.

        The API no longer accepts Style classes. Callers must pass an instantiated
        Style (e.g. ``ff.def_style(AtomStyle("full"))``). If a style with the
        same runtime class and name already exists it will be returned instead of
        registering a duplicate.
        """
        if not isinstance(style, Style):
            raise TypeError(
                "def_style expects a Style instance; passing a class is no longer supported"
            )

        style_inst: Style = style
        style_cls = style_inst.__class__
        style_name = style_inst.name

        # Return existing style if one with same class/name exists
        for s in self.styles.bucket(style_cls):
            if s.name == style_name:
                return s

        # Otherwise register provided instance
        self.styles.add(style_inst)
        return style_inst

    def get_styles(self, style_class: type[S]) -> List[S]:
        return cast(List[S], self.styles.bucket(style_class))

    def get_types(self, type_class: type[Ty]) -> list[Ty]:
        all_types = set()
        for style in self.styles.bucket(Style):
            all_types.update(style.types.bucket(type_class))
        return list(all_types)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def merge(self, other: "ForceField"):
        for other_style in other.styles.bucket(Style):
            style_bucket = self.styles.bucket(type(other_style))
            found = False
            for style in style_bucket:
                if style.name == other_style.name:
                    style.merge(other_style)
                    found = True
                    break
            if not found:
                self.styles.add(other_style)
        return self

    def to_potentials(self):
        """Create Potential instances from all styles in ForceField.

        Returns:
            Potentials collection containing all created potential instances

        Note:
            Only Styles that support to_potential() method will be converted (e.g. BondStyle, AngleStyle, PairStyle)
        """
        # Delayed import to avoid circular references
        from molpy.potential.base import Potentials

        potentials = Potentials()

        # Iterate over all styles and try to create corresponding potentials
        for style in self.styles.bucket(Style):
            # Check if style has to_potential method
            if hasattr(style, "to_potential"):
                try:
                    # mypy cannot infer that 'style' has to_potential, so cast
                    potential = cast(Any, style).to_potential()
                    if potential is not None:
                        potentials.append(potential)
                except (ValueError, AttributeError):
                    # Skip if creation fails (e.g. missing parameters or Potential class not found)
                    # Could log warnings, but silently skip for now
                    pass

        return potentials


# ===================================================================
#               Extended AtomisticForcefield class
# ===================================================================


class AtomType(Type):
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class BondType(Type):
    """Bond type defined by two atom types"""

    def __init__(self, name: str, itom: "AtomType", jtom: "AtomType", **kwargs: Any):
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom

    def matches(self, at1: "AtomType", at2: "AtomType") -> bool:
        """Check if matches given atom type pair (supports wildcards and order-independent)"""
        return (self.itom == at1 and self.jtom == at2) or (
            self.itom == at2 and self.jtom == at1
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}>"


class AngleType(Type):
    """Angle type defined by three atom types"""

    def __init__(
        self,
        name: str,
        itom: "AtomType",
        jtom: "AtomType",
        ktom: "AtomType",
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom

    def matches(self, at1: "AtomType", at2: "AtomType", at3: "AtomType") -> bool:
        """Check if matches given atom type triple (supports wildcards and reverse order)"""
        # Forward match
        if self.itom == at1 and self.jtom == at2 and self.ktom == at3:
            return True
        # Reverse match
        return bool(self.itom == at3 and self.jtom == at2 and self.ktom == at1)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}-{self.ktom.name}>"


class DihedralType(Type):
    """Dihedral type defined by four atom types"""

    def __init__(
        self,
        name: str,
        itom: "AtomType",
        jtom: "AtomType",
        ktom: "AtomType",
        ltom: "AtomType",
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def matches(
        self, at1: "AtomType", at2: "AtomType", at3: "AtomType", at4: "AtomType"
    ) -> bool:
        """Check if matches given atom type quadruple (supports wildcards and reverse order)"""
        # Forward match
        if (
            self.itom == at1
            and self.jtom == at2
            and self.ktom == at3
            and self.ltom == at4
        ):
            return True
        # Reverse match
        return bool(
            self.itom == at4
            and self.jtom == at3
            and self.ktom == at2
            and self.ltom == at1
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}-{self.ktom.name}-{self.ltom.name}>"


class ImproperType(Type):
    """Improper dihedral type defined by four atom types"""

    def __init__(
        self,
        name: str,
        itom: "AtomType",
        jtom: "AtomType",
        ktom: "AtomType",
        ltom: "AtomType",
        **kwargs: Any,
    ):
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def matches(
        self, at1: "AtomType", at2: "AtomType", at3: "AtomType", at4: "AtomType"
    ) -> bool:
        """Check if matches given atom type quadruple (supports wildcards)"""
        # Improper typically has specific central atom, so matching rules may differ
        # Implement simple exact matching for now
        return (
            self.itom == at1
            and self.jtom == at2
            and self.ktom == at3
            and self.ltom == at4
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}-{self.ktom.name}-{self.ltom.name}>"


class PairType(Type):
    """Non-bonded interaction type defined by one or two atom types"""

    def __init__(self, name: str, *atom_types: "AtomType", **kwargs: Any):
        super().__init__(name, **kwargs)
        if len(atom_types) == 1:
            self.itom = atom_types[0]
            self.jtom = atom_types[0]  # Self-interaction
        elif len(atom_types) == 2:
            self.itom = atom_types[0]
            self.jtom = atom_types[1]
        else:
            raise ValueError("PairType requires 1 or 2 atom types")

    def matches(self, at1: "AtomType", at2: "AtomType | None" = None) -> bool:
        """Check if matches given atom type pair (supports wildcards and order-independent)"""
        if at2 is None:
            at2 = at1  # Self-interaction

        return (self.itom == at1 and self.jtom == at2) or (
            self.itom == at2 and self.jtom == at1
        )

    def __repr__(self) -> str:
        if self.itom == self.jtom:
            return f"<{self.__class__.__name__}: {self.itom.name}>"
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}>"


class AtomStyle(Style):
    def def_type(self, name: str, **kwargs: Any) -> AtomType:
        """Define atom type

        Args:
            type_: Specific type identifier (e.g. opls_135)
            class_: Class identifier (e.g. CT)
            **kwargs: Other parameters (element, mass, etc.)

        Returns:
            Created AtomType instance
        """
        at = AtomType(name=name, **kwargs)
        self.types.add(at)
        return at


class BondStyle(Style):
    def def_type(
        self, itom: AtomType, jtom: AtomType, name: str = "", **kwargs: Any
    ) -> BondType:
        """Define bond type

        Args:
            itom: First atom type
            jtom: Second atom type
            name: Optional name (defaults to itom-jtom)
            **kwargs: Bond parameters (e.g. k, r0, etc.)
        """
        if not name:
            name = f"{itom.name}-{jtom.name}"
        bt = BondType(name, itom, jtom, **kwargs)
        self.types.add(bt)
        return bt

    def to_potential(self):
        """Create corresponding Potential instance from BondStyle.

        Returns:
            Potential instance containing all BondType parameters

        Raises:
            ValueError: If corresponding Potential class not found or missing required parameters
        """
        # Delayed import to avoid circular references

        # Get corresponding Potential class from registry
        typename = "bond"
        registry = ForceField._kernel_registry.get(typename, {})
        potential_class = registry.get(self.name)

        if potential_class is None:
            raise ValueError(
                f"Potential class not found for bond style '{self.name}'. "
                f"Available potentials: {list(registry.keys())}"
            )

        # Get all BondTypes
        bond_types = self.types.bucket(BondType)
        if not bond_types:
            raise ValueError(f"No bond types defined in style '{self.name}'")

        # Extract parameters
        k_list = []
        r0_list = []

        for bt in bond_types:
            k = bt.params.kwargs.get("k")
            r0 = bt.params.kwargs.get("r0")

            if k is None or r0 is None:
                raise ValueError(
                    f"BondType '{bt.name}' is missing required parameters: "
                    f"k={k}, r0={r0}"
                )

            k_list.append(k)
            r0_list.append(r0)

        # Create Potential instance
        import numpy as np

        return potential_class(k=np.array(k_list), r0=np.array(r0_list))


class AngleStyle(Style):
    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        name: str = "",
        **kwargs: Any,
    ) -> AngleType:
        """Define angle type

        Args:
            itom: First atom type
            jtom: Central atom type
            ktom: Third atom type
            name: Optional name (defaults to itom-jtom-ktom)
            **kwargs: Angle parameters (e.g. k, theta0, etc.)
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}"
        at = AngleType(name, itom, jtom, ktom, **kwargs)
        self.types.add(at)
        return at

    def to_potential(self):
        """Create corresponding Potential instance from AngleStyle.

        Returns:
            Potential instance containing all AngleType parameters

        Raises:
            ValueError: If corresponding Potential class not found or missing required parameters
        """
        # Delayed import to avoid circular references

        # Get corresponding Potential class from registry
        typename = "angle"
        registry = ForceField._kernel_registry.get(typename, {})
        potential_class = registry.get(self.name)

        if potential_class is None:
            raise ValueError(
                f"Potential class not found for angle style '{self.name}'. "
                f"Available potentials: {list(registry.keys())}"
            )

        # Get all AngleTypes
        angle_types = self.types.bucket(AngleType)
        if not angle_types:
            raise ValueError(f"No angle types defined in style '{self.name}'")

        # Extract parameters
        k_list = []
        theta0_list = []

        for at in angle_types:
            k = at.params.kwargs.get("k")
            theta0 = at.params.kwargs.get("theta0")

            if k is None or theta0 is None:
                raise ValueError(
                    f"AngleType '{at.name}' is missing required parameters: "
                    f"k={k}, theta0={theta0}"
                )

            k_list.append(k)
            theta0_list.append(theta0)

        # Create Potential instance
        import numpy as np

        return potential_class(k=np.array(k_list), theta0=np.array(theta0_list))


class DihedralStyle(Style):
    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        name: str = "",
        **kwargs: Any,
    ) -> DihedralType:
        """Define dihedral type

        Args:
            itom: First atom type
            jtom: Second atom type
            ktom: Third atom type
            ltom: Fourth atom type
            name: Optional name (defaults to itom-jtom-ktom-ltom)
            **kwargs: Dihedral parameters
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralType(name, itom, jtom, ktom, ltom, **kwargs)
        self.types.add(dt)
        return dt


class ImproperStyle(Style):
    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        name: str = "",
        **kwargs: Any,
    ) -> ImproperType:
        """Define improper dihedral type

        Args:
            itom: First atom type
            jtom: Second atom type (usually central atom)
            ktom: Third atom type
            ltom: Fourth atom type
            name: Optional name (defaults to itom-jtom-ktom-ltom)
            **kwargs: Improper dihedral parameters
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        it = ImproperType(name, itom, jtom, ktom, ltom, **kwargs)
        self.types.add(it)
        return it


class PairStyle(Style):
    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType | None = None,
        name: str = "",
        **kwargs: Any,
    ) -> PairType:
        """Define non-bonded interaction type

        Args:
            itom: First atom type
            jtom: Second atom type (optional, defaults to same as itom for self-interaction)
            name: Optional name
            **kwargs: Non-bonded parameters (e.g. sigma, epsilon, charge, etc.)
        """
        if jtom is None:
            jtom = itom

        if not name:
            name = itom.name if itom == jtom else f"{itom.name}-{jtom.name}"

        pt = PairType(name, itom, jtom, **kwargs)
        self.types.add(pt)
        return pt

    def to_potential(self):
        """Create corresponding Potential instance from PairStyle.

        Returns:
            Potential instance containing all PairType parameters

        Raises:
            ValueError: If corresponding Potential class not found or missing required parameters
        """
        # Delayed import to avoid circular references

        # Get corresponding Potential class from registry
        typename = "pair"
        registry = ForceField._kernel_registry.get(typename, {})
        potential_class = registry.get(self.name)

        if potential_class is None:
            raise ValueError(
                f"Potential class not found for pair style '{self.name}'. "
                f"Available potentials: {list(registry.keys())}"
            )

        # Get all PairTypes
        pair_types = self.types.bucket(PairType)
        if not pair_types:
            raise ValueError(f"No pair types defined in style '{self.name}'")

        # Extract parameters
        epsilon_list = []
        sigma_list = []

        for pt in pair_types:
            epsilon = pt.params.kwargs.get("epsilon")
            sigma = pt.params.kwargs.get("sigma")

            if epsilon is None or sigma is None:
                raise ValueError(
                    f"PairType '{pt.name}' is missing required parameters: "
                    f"epsilon={epsilon}, sigma={sigma}"
                )

            epsilon_list.append(epsilon)
            sigma_list.append(sigma)

        # Create Potential instance
        import numpy as np

        return potential_class(
            epsilon=np.array(epsilon_list), sigma=np.array(sigma_list)
        )


class AtomisticForcefield(ForceField):
    def def_atomstyle(self, name: str, *args: Any, **kwargs: Any) -> AtomStyle:
        return cast(AtomStyle, self.def_style(AtomStyle(name, *args, **kwargs)))

    def def_bondstyle(self, name: str, *args: Any, **kwargs: Any) -> BondStyle:
        return cast(BondStyle, self.def_style(BondStyle(name, *args, **kwargs)))

    def def_anglestyle(self, name: str, *args: Any, **kwargs: Any) -> AngleStyle:
        return cast(AngleStyle, self.def_style(AngleStyle(name, *args, **kwargs)))

    def def_dihedralstyle(self, name: str, *args: Any, **kwargs: Any) -> DihedralStyle:
        return cast(DihedralStyle, self.def_style(DihedralStyle(name, *args, **kwargs)))

    def def_improperstyle(self, name: str, *args: Any, **kwargs: Any) -> ImproperStyle:
        return cast(ImproperStyle, self.def_style(ImproperStyle(name, *args, **kwargs)))

    def def_pairstyle(self, name: str, *args: Any, **kwargs: Any) -> PairStyle:
        return cast(PairStyle, self.def_style(PairStyle(name, *args, **kwargs)))

    def get_atomtypes(self) -> list[AtomType]:
        return self.get_types(AtomType)

    def get_bondtypes(self) -> list[BondType]:
        return self.get_types(BondType)

    def get_angletypes(self) -> list[AngleType]:
        return self.get_types(AngleType)
