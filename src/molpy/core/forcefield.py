from collections import defaultdict
from collections.abc import Iterator
from typing import Any, List, Self, TypeVar, cast
import numpy as np

from .entity import Entity

# --- Generic variables and TypeBucket implementation ---

T = TypeVar("T")
S = TypeVar("S", bound="Style")
Ty = TypeVar("Ty", bound="Type")


def get_nearest_type[T](item: T) -> type[T]:
    """Return the most specific runtime type of an item.

    Args:
        item: Any object whose runtime class is needed.

    Returns:
        The concrete ``type`` of *item*.

    Related:
        TypeBucket: Uses this to partition items by their runtime type.
    """
    return type(item)


class TypeBucket[T]:
    """A collection that partitions items into buckets keyed by their runtime type.

    Items are stored in sets grouped by ``type(item)``.  Retrieval via
    ``bucket(cls)`` returns all items whose runtime type is a subclass of
    *cls*, enabling polymorphic queries over heterogeneous collections.

    Related:
        Style: Uses TypeBucket to store Type instances.
        ForceField: Uses TypeBucket to store Style instances.
    """

    def __init__(self) -> None:
        self._items: dict[type[T], set[T]] = defaultdict(set)

    def add(self, item: T) -> None:
        """Add an item to the bucket corresponding to its runtime type.

        Args:
            item: The item to store.  It is placed in the set keyed by
                ``type(item)``.
        """
        cls = get_nearest_type(item)
        self._items[cls].add(item)

    def remove(self, item: T) -> None:
        """Remove an item from its type bucket.

        If the item is not present, this is a no-op (uses ``discard``).

        Args:
            item: The item to remove.
        """
        cls = get_nearest_type(item)
        self._items[cls].discard(item)

    def bucket(self, cls: type) -> List[T]:
        """Retrieve all items whose runtime type is a subclass of *cls*.

        Args:
            cls: The base class to match against.  All items stored under
                a key ``k`` where ``issubclass(k, cls)`` is ``True`` are
                included.

        Returns:
            A flat list of matching items (order is not guaranteed).
        """
        result: List[T] = []
        for k, items in self._items.items():
            # runtime check: k is a class, cls is a class
            if issubclass(k, cls):
                result.extend(list(items))
        return result

    def classes(self) -> Iterator[type[T]]:
        """Iterate over the distinct runtime types that have been stored.

        Returns:
            An iterator of ``type`` objects representing all buckets that
            contain at least one item.
        """
        return iter(self._items.keys())


# --- Basic components ---


class Parameters:
    """Container for positional and keyword parameters of a force-field type or style.

    Provides indexed access to positional arguments (by ``int`` or ``slice``)
    and keyword arguments (by ``str``).

    Related:
        Type: Stores a Parameters instance as ``params``.
        Style: Stores a Parameters instance as ``params``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialise with arbitrary positional and keyword arguments.

        Args:
            *args: Positional parameters (stored as a list).
            **kwargs: Keyword parameters (stored as a dict).
        """
        self.args = list(args)
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return f"Parameters(args={self.args}, kwargs={self.kwargs})"

    def __getitem__(self, key: int | slice | str):
        """Retrieve a parameter by index or name.

        Args:
            key: An ``int`` or ``slice`` for positional lookup, or a ``str``
                for keyword lookup.

        Returns:
            The corresponding parameter value.

        Raises:
            KeyError: If *key* is a ``str`` not present in keyword arguments.
            IndexError: If *key* is an ``int`` outside the positional range.
        """
        if isinstance(key, str):
            return self.kwargs[key]
        else:
            return self.args[key]


class Type:
    """Base class for all force-field type descriptors.

    A Type carries a unique *name* together with arbitrary ``Parameters``
    that describe a particular force-field entry (e.g. atom type, bond type).
    Types are hashed and compared by ``(class, name)`` so two instances of
    the same subclass with the same name are considered equal.

    Related:
        AtomType, BondType, AngleType, DihedralType, ImproperType, PairType:
            Concrete subclasses for specific interaction categories.
        Style: Groups related Type instances under a named interaction style.
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        """Initialise a force-field type.

        Args:
            name: Unique identifier for this type (e.g. ``"opls_135"``).
            *args: Additional positional parameters forwarded to ``Parameters``.
            **kwargs: Keyword parameters forwarded to ``Parameters`` (e.g.
                ``mass=12.011``, ``charge=-0.18``).
        """
        self._name = name
        self.params = Parameters(*args, **kwargs)

    @property
    def name(self) -> str:
        """The unique name identifying this type."""
        return self._name

    def __hash__(self) -> int:
        return hash((self.__class__, self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on class and name.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` if *other* is a ``Type`` of the same class with the
            same name.
        """
        if not isinstance(other, Type):
            return False
        return self.__class__ == other.__class__ and self.name == other.name

    def __gt__(self, other: object) -> bool:
        """Lexicographic ordering by name within the same class.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` if *other* is a ``Type`` of the same class and
            ``self.name > other.name``.
        """
        if not isinstance(other, Type):
            return False
        return self.__class__ == other.__class__ and self.name > other.name

    def __getitem__(self, key: str) -> Any:
        """Look up a keyword parameter by name.

        Args:
            key: Parameter name.

        Returns:
            The parameter value, or ``None`` if not present.
        """
        return self.params.kwargs.get(key, None)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a keyword parameter.

        Args:
            key: Parameter name.
            value: Parameter value.
        """
        self.params.kwargs[key] = value

    def __contains__(self, key: str) -> bool:
        """Check whether a keyword parameter exists.

        Args:
            key: Parameter name to look up.

        Returns:
            ``True`` if the parameter is present.
        """
        return key in self.params.kwargs

    def get(self, key: str, default: Any = None) -> Any:
        """Return a keyword parameter or a default value.

        Args:
            key: Parameter name.
            default: Value returned when *key* is absent. Defaults to ``None``.

        Returns:
            The parameter value if present, otherwise *default*.
        """
        return self.params.kwargs.get(key, default)

    def copy(self) -> "Type":
        """Create a copy of this type with the same name and parameters.

        Returns:
            A new Type instance with copied parameters
        """
        # Get the actual type class
        actual_type_class = type(self)

        # Copy parameters
        type_params = self.params.kwargs.copy()
        type_args = list(self.params.args)

        # Handle special types with atom type references
        if isinstance(self, BondType):
            # BondType requires itom and jtom as positional args
            return actual_type_class(
                self.name, self.itom, self.jtom, *type_args, **type_params
            )
        elif isinstance(self, AngleType):
            # AngleType requires itom, jtom, ktom as positional args
            return actual_type_class(
                self.name, self.itom, self.jtom, self.ktom, *type_args, **type_params
            )
        elif isinstance(self, DihedralType):
            # DihedralType requires itom, jtom, ktom, ltom as positional args
            return actual_type_class(
                self.name,
                self.itom,
                self.jtom,
                self.ktom,
                self.ltom,
                *type_args,
                **type_params,
            )
        elif isinstance(self, ImproperType):
            # ImproperType requires itom, jtom, ktom, ltom as positional args
            return actual_type_class(
                self.name,
                self.itom,
                self.jtom,
                self.ktom,
                self.ltom,
                *type_args,
                **type_params,
            )
        elif isinstance(self, PairType):
            # PairType requires atom types as positional args
            if self.itom == self.jtom:
                return actual_type_class(
                    self.name, self.itom, *type_args, **type_params
                )
            else:
                return actual_type_class(
                    self.name, self.itom, self.jtom, *type_args, **type_params
                )
        else:
            # Regular Type (e.g., AtomType) - just name and kwargs
            return actual_type_class(self.name, *type_args, **type_params)


class Style:
    """Named interaction style that groups related Type instances.

    A Style represents a particular functional form for an interaction
    category (e.g. ``BondStyle("harmonic")``).  It holds a ``TypeBucket``
    of ``Type`` instances that share the same functional form.

    Related:
        Type: Individual type entries stored within a Style.
        ForceField: Aggregates multiple Style instances.
    """

    def __init__(self, name: str, *args: Any, **kwargs: Any):
        """Initialise a named interaction style.

        Args:
            name: Name of the interaction style (e.g. ``"harmonic"``,
                ``"lj/cut"``).
            *args: Additional positional parameters forwarded to
                ``Parameters``.
            **kwargs: Keyword parameters forwarded to ``Parameters``.
        """
        self.name = name
        self.params = Parameters(*args, **kwargs)
        self.types = TypeBucket[Type]()

    def __hash__(self) -> int:
        return hash((self.__class__, self.name))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __eq__(self, other: object) -> bool:
        """Check equality based on class and name.

        Args:
            other: Object to compare against.

        Returns:
            ``True`` if *other* is the same Style subclass with the same name.
        """
        return isinstance(other, self.__class__) and self.name == other.name

    def merge(self, other: "Style") -> "Style":
        """Merge another Style's parameters and types into this one.

        Parameters from *other* are appended (positional) or updated
        (keyword).  All types from *other* are added to this style's
        type bucket.

        Args:
            other: The style whose contents are merged in.

        Returns:
            This style instance (for chaining).
        """
        if not self.params.args:
            self.params.args.extend(other.params.args)
        self.params.kwargs.update(other.params.kwargs)
        for t in other.types.bucket(Type):
            self.types.add(t)
        return self

    def copy(self) -> "Style":
        """Create a copy of this style with the same name and parameters (but not types).

        Returns:
            A new Style instance with copied name and parameters
        """
        import inspect

        # Get the actual style class
        actual_style_class = type(self)

        # Get constructor signature to determine how to create the copy
        sig = inspect.signature(actual_style_class.__init__)
        param_count = len(sig.parameters) - 1  # Exclude 'self'

        # Copy parameters
        style_params = self.params.kwargs.copy()
        style_args = list(self.params.args)

        if param_count == 0:
            # Style with no parameters (shouldn't happen for base Style, but handle it)
            new_style = actual_style_class(self.name)
        elif param_count == 1:
            # Style with just name parameter
            new_style = actual_style_class(self.name)
        else:
            # Style with name and additional parameters
            new_style = actual_style_class(self.name, *style_args, **style_params)

        return new_style

    def get_types(self, type_class: type[Ty]) -> list[Ty]:
        """Get all types of the specified class from this style.

        Args:
            type_class: Class of the types to retrieve (e.g., AtomType, BondType)

        Returns:
            List of types of the specified class
        """
        return cast(list[Ty], self.types.bucket(type_class))

    def get_type_by_name(self, name: str, type_class: type[Ty] = Type) -> Ty | None:
        """Get a type by name from this style.

        Args:
            name: Name of the type to find
            type_class: Class of the type to search for (defaults to Type)

        Returns:
            The first matching Type instance, or None if not found
        """
        for type_obj in self.types.bucket(type_class):
            if type_obj.name == name:
                return cast(Ty, type_obj)
        return None


# ===================================================================
#                    ForceField base class
# ===================================================================


class ForceField:
    """Top-level container for a complete force-field definition.

    A ForceField aggregates multiple ``Style`` instances (atom, bond,
    angle, dihedral, improper, pair) and provides lookup, merging, and
    conversion to ``Potential`` objects for energy evaluation.

    Attributes:
        name: Human-readable name of the force field.
        units: Unit system used by the parameters (e.g. ``"real"``,
            ``"metal"``, ``"lj"``).

    Related:
        Style: Interaction styles stored within the force field.
        AtomisticForcefield: Convenience subclass with shorthand methods.
    """

    # Kernel registry for potential functions
    _kernel_registry: dict[str, dict[str, type]] = {}

    def __init__(self, name: str = "", units: str = "real"):
        """Initialise a force field.

        Args:
            name: Name of the force field (e.g. ``"OPLS-AA"``).
            units: Unit convention for parameter values.  Defaults to
                ``"real"`` (kcal/mol, Angstrom).
        """
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
        """Retrieve all styles that are instances of the given class.

        Args:
            style_class: The Style subclass to filter by (e.g.
                ``BondStyle``).

        Returns:
            List of matching Style instances.
        """
        return cast(List[S], self.styles.bucket(style_class))

    def get_style_by_name(self, name: str, style_class: type[S] = Style) -> S | None:
        """Get a style by name from the force field.

        Args:
            name: Name of the style to find
            style_class: Class of the style to search for (defaults to Style)

        Returns:
            The first matching Style instance, or None if not found
        """
        for style in self.styles.bucket(style_class):
            if style.name == name:
                return cast(S, style)
        return None

    def get_types(self, type_class: type[Ty]) -> list[Ty]:
        """Collect all types of the given class from every registered style.

        Args:
            type_class: The Type subclass to collect (e.g. ``AtomType``).

        Returns:
            Deduplicated list of matching Type instances across all styles.
        """
        all_types = set()
        for style in self.styles.bucket(Style):
            all_types.update(style.types.bucket(type_class))
        return list(all_types)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def merge(self, other: "ForceField") -> "ForceField":
        """Merge another force field's styles and types into this one.

        For each style in *other*, if a style with the same class and name
        already exists in this force field it is merged in-place; otherwise
        the style is added as-is.

        Args:
            other: The force field whose contents are merged in.

        Returns:
            This force field instance (for chaining).
        """
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

    def rename_type(self, style_class: type[S], old: str, new: str) -> int:
        """Rename every Type named ``old`` to ``new`` in matching styles.

        After mutating the internal ``_name`` the type is removed and
        re-added to its bucket so set-based hash lookups stay consistent.

        Args:
            style_class: Style subclass whose type bucket is targeted.
            old: Existing type name.
            new: Replacement type name.

        Returns:
            Number of Types renamed.
        """
        count = 0
        for style in self.styles.bucket(style_class):
            bucket = style.types
            targets = [t for t in bucket.bucket(Type) if t.name == old]
            for t in targets:
                bucket.remove(t)
                t._name = new
                bucket.add(t)
                count += 1
        return count

    def remove_type(self, style_class: type[S], name: str) -> int:
        """Remove every Type named ``name`` from styles of ``style_class``.

        Returns the number of Types removed.
        """
        count = 0
        for style in self.styles.bucket(style_class):
            bucket = style.types
            targets = [t for t in bucket.bucket(Type) if t.name == name]
            for t in targets:
                bucket.remove(t)
                count += 1
        return count

    def remove_style(self, style_class: type[S], name: str) -> bool:
        """Remove a Style instance of ``style_class`` whose name matches.

        Returns ``True`` if a style was removed.
        """
        for style in list(self.styles.bucket(style_class)):
            if style.name == name:
                self.styles.remove(style)
                return True
        return False

    def to_potentials(self) -> "Potentials":
        """Create Potential instances from all styles in ForceField.

        Returns:
            Potentials: Collection containing all created potential instances.

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
    """Force-field type for a single atom.

    An AtomType carries parameters such as mass, charge, and element
    symbol.  Typical keyword parameters include:

    * ``mass`` -- atomic mass in g/mol (amu)
    * ``charge`` -- partial charge in elementary charge units (e)
    * ``element`` -- chemical element symbol

    Related:
        AtomStyle: Style that groups AtomType instances.
        BondType: References AtomType instances to define bond endpoints.
    """

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"


class BondType(Type):
    """Bond type defined by two atom types"""

    def __init__(self, name: str, itom: "AtomType", jtom: "AtomType", **kwargs: Any):
        """Initialise a bond type between two atom types.

        Args:
            name: Unique identifier for this bond type.
            itom: First endpoint atom type.
            jtom: Second endpoint atom type.
            **kwargs: Bond parameters (e.g. ``k`` in kcal/(mol*A^2),
                ``r0`` in Angstrom).
        """
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom

    def matches(self, at1: "AtomType", at2: "AtomType") -> bool:
        """Check whether this bond type matches an atom type pair.

        Matching is order-independent: ``(at1, at2)`` matches if
        ``(itom, jtom)`` equals ``(at1, at2)`` or ``(at2, at1)``.

        Args:
            at1: First atom type.
            at2: Second atom type.

        Returns:
            ``True`` if the pair matches this bond type.
        """
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
        """Initialise an angle type defined by three atom types.

        Args:
            name: Unique identifier for this angle type.
            itom: First endpoint atom type.
            jtom: Central (vertex) atom type.
            ktom: Third endpoint atom type.
            **kwargs: Angle parameters (e.g. ``k`` in kcal/(mol*rad^2),
                ``theta0`` in radians).
        """
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom

    def matches(self, at1: "AtomType", at2: "AtomType", at3: "AtomType") -> bool:
        """Check whether this angle type matches an atom type triple.

        Matching considers both forward ``(at1, at2, at3)`` and reverse
        ``(at3, at2, at1)`` orderings; the central atom must always
        match ``jtom``.

        Args:
            at1: First endpoint atom type.
            at2: Central atom type.
            at3: Third endpoint atom type.

        Returns:
            ``True`` if the triple matches this angle type.
        """
        # Forward match
        if self.itom == at1 and self.jtom == at2 and self.ktom == at3:
            return True
        # Reverse match
        return bool(self.itom == at3 and self.jtom == at2 and self.ktom == at1)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.itom.name}-{self.jtom.name}-{self.ktom.name}>"


class DihedralType(Type):
    """Dihedral (torsion) type defined by four atom types.

    Related:
        DihedralStyle: Style that groups DihedralType instances.
    """

    def __init__(
        self,
        name: str,
        itom: "AtomType",
        jtom: "AtomType",
        ktom: "AtomType",
        ltom: "AtomType",
        **kwargs: Any,
    ):
        """Initialise a dihedral type for four atom types.

        Args:
            name: Unique identifier for this dihedral type.
            itom: First atom type.
            jtom: Second atom type.
            ktom: Third atom type.
            ltom: Fourth atom type.
            **kwargs: Dihedral parameters (e.g. ``k`` in kcal/mol,
                ``d`` sign factor, ``n`` multiplicity).
        """
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def matches(
        self, at1: "AtomType", at2: "AtomType", at3: "AtomType", at4: "AtomType"
    ) -> bool:
        """Check whether this dihedral type matches an atom type quadruple.

        Both forward ``(at1, at2, at3, at4)`` and reverse
        ``(at4, at3, at2, at1)`` orderings are considered equivalent.

        Args:
            at1: First atom type.
            at2: Second atom type.
            at3: Third atom type.
            at4: Fourth atom type.

        Returns:
            ``True`` if the quadruple matches this dihedral type.
        """
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
    """Improper dihedral type defined by four atom types.

    Improper dihedrals typically have a designated central atom (often
    ``jtom``) and are used to enforce planarity or chirality.

    Related:
        ImproperStyle: Style that groups ImproperType instances.
    """

    def __init__(
        self,
        name: str,
        itom: "AtomType",
        jtom: "AtomType",
        ktom: "AtomType",
        ltom: "AtomType",
        **kwargs: Any,
    ):
        """Initialise an improper dihedral type for four atom types.

        Args:
            name: Unique identifier for this improper type.
            itom: First atom type.
            jtom: Second atom type (typically the central atom).
            ktom: Third atom type.
            ltom: Fourth atom type.
            **kwargs: Improper parameters (e.g. ``k`` in kcal/mol,
                ``d`` sign factor, ``n`` multiplicity).
        """
        super().__init__(name, **kwargs)
        self.itom = itom
        self.jtom = jtom
        self.ktom = ktom
        self.ltom = ltom

    def matches(
        self, at1: "AtomType", at2: "AtomType", at3: "AtomType", at4: "AtomType"
    ) -> bool:
        """Check whether this improper type matches an atom type quadruple.

        Uses exact positional matching (no reverse ordering) because
        improper dihedrals have a specific central atom convention.

        Args:
            at1: First atom type.
            at2: Second atom type.
            at3: Third atom type.
            at4: Fourth atom type.

        Returns:
            ``True`` if the quadruple matches this improper type exactly.
        """
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
        """Initialise a non-bonded pair type.

        Args:
            name: Unique identifier for this pair type.
            *atom_types: One or two ``AtomType`` instances.  A single
                type implies a self-interaction.
            **kwargs: Pair parameters (e.g. ``epsilon`` in kcal/mol,
                ``sigma`` in Angstrom).

        Raises:
            ValueError: If not exactly 1 or 2 atom types are provided.
        """
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
        """Check whether this pair type matches an atom type pair.

        Matching is order-independent.  If only *at1* is given, it is
        treated as a self-interaction query.

        Args:
            at1: First atom type.
            at2: Second atom type.  Defaults to *at1* (self-interaction).

        Returns:
            ``True`` if the pair matches this pair type.
        """
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
    """Style for atom types (e.g. ``"full"``, ``"charge"``).

    Related:
        AtomType: Type instances managed by this style.
        AtomisticForcefield.def_atomstyle: Shorthand factory.
    """

    def def_type(self, name: str, **kwargs: Any) -> AtomType:
        """Define atom type.

        Args:
            name (str): Name for the atom type.
            **kwargs (Any): Other parameters (element, mass, etc.).

        Returns:
            AtomType: Created AtomType instance.
        """
        at = AtomType(name=name, **kwargs)
        self.types.add(at)
        return at


class BondStyle(Style):
    """Style for bond interactions (e.g. ``"harmonic"``).

    Related:
        BondType: Type instances managed by this style.
        AtomisticForcefield.def_bondstyle: Shorthand factory.
    """

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

    def to_potential(self) -> "Potential":
        """Create corresponding Potential instance from BondStyle.

        Returns:
            Potential: Potential instance that accepts string type labels (from Frame).
                The potential internally uses dictionaries to map type names to parameters.

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

        # Extract parameters as dictionaries (type name -> parameter)
        k_dict = {}
        r0_dict = {}

        for bt in bond_types:
            k = bt.params.kwargs.get("k")
            r0 = bt.params.kwargs.get("r0")

            if k is None or r0 is None:
                raise ValueError(
                    f"BondType '{bt.name}' is missing required parameters: "
                    f"k={k}, r0={r0}"
                )

            k_dict[bt.name] = k
            r0_dict[bt.name] = r0

        # Create Potential instance with dictionaries
        # TypeIndexedArray automatically handles string type name indexing
        return potential_class(k=k_dict, r0=r0_dict)


class AngleStyle(Style):
    """Style for angle interactions (e.g. ``"harmonic"``).

    Related:
        AngleType: Type instances managed by this style.
        AtomisticForcefield.def_anglestyle: Shorthand factory.
    """

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

    def to_potential(self) -> "Potential":
        """Create corresponding Potential instance from AngleStyle.

        Returns:
            Potential: Potential instance that accepts string type labels (from Frame).
                The potential internally uses TypeIndexedArray to map type names to parameters.

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

        # Extract parameters as dictionaries (type name -> parameter)
        k_dict = {}
        theta0_dict = {}

        for at in angle_types:
            k = at.params.kwargs.get("k")
            theta0 = at.params.kwargs.get("theta0")

            if k is None or theta0 is None:
                raise ValueError(
                    f"AngleType '{at.name}' is missing required parameters: "
                    f"k={k}, theta0={theta0}"
                )

            k_dict[at.name] = k
            theta0_dict[at.name] = theta0

        # Create Potential instance with dictionaries
        # TypeIndexedArray automatically handles string type name indexing
        return potential_class(k=k_dict, theta0=theta0_dict)


class DihedralStyle(Style):
    """Style for dihedral (torsion) interactions (e.g. ``"opls"``).

    Related:
        DihedralType: Type instances managed by this style.
        AtomisticForcefield.def_dihedralstyle: Shorthand factory.
    """

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
    """Style for improper dihedral interactions (e.g. ``"cvff"``).

    Related:
        ImproperType: Type instances managed by this style.
        AtomisticForcefield.def_improperstyle: Shorthand factory.
    """

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
    """Style for non-bonded pair interactions (e.g. ``"lj/cut"``).

    Related:
        PairType: Type instances managed by this style.
        AtomisticForcefield.def_pairstyle: Shorthand factory.
    """

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

    def to_potential(self) -> "Potential":
        """Create corresponding Potential instance from PairStyle.

        Returns:
            Potential: Potential instance containing all PairType parameters.

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
    """Convenience subclass of ForceField with shorthand methods for atomistic styles.

    Provides ``def_*style`` helpers that create and register the
    appropriate Style subclass in a single call, and ``get_*types``
    accessors for the most common interaction categories.

    Related:
        ForceField: Base class with generic style/type management.
    """

    def def_atomstyle(self, name: str, *args: Any, **kwargs: Any) -> AtomStyle:
        """Create and register an AtomStyle.

        Args:
            name: Name of the atom style (e.g. ``"full"``).
            *args: Positional parameters forwarded to AtomStyle.
            **kwargs: Keyword parameters forwarded to AtomStyle.

        Returns:
            The registered (or existing) AtomStyle instance.
        """
        return cast(AtomStyle, self.def_style(AtomStyle(name, *args, **kwargs)))

    def def_bondstyle(self, name: str, *args: Any, **kwargs: Any) -> BondStyle:
        """Create and register a BondStyle.

        Args:
            name: Name of the bond style (e.g. ``"harmonic"``).
            *args: Positional parameters forwarded to BondStyle.
            **kwargs: Keyword parameters forwarded to BondStyle.

        Returns:
            The registered (or existing) BondStyle instance.
        """
        return cast(BondStyle, self.def_style(BondStyle(name, *args, **kwargs)))

    def def_anglestyle(self, name: str, *args: Any, **kwargs: Any) -> AngleStyle:
        """Create and register an AngleStyle.

        Args:
            name: Name of the angle style (e.g. ``"harmonic"``).
            *args: Positional parameters forwarded to AngleStyle.
            **kwargs: Keyword parameters forwarded to AngleStyle.

        Returns:
            The registered (or existing) AngleStyle instance.
        """
        return cast(AngleStyle, self.def_style(AngleStyle(name, *args, **kwargs)))

    def def_dihedralstyle(self, name: str, *args: Any, **kwargs: Any) -> DihedralStyle:
        """Create and register a DihedralStyle.

        Args:
            name: Name of the dihedral style (e.g. ``"opls"``).
            *args: Positional parameters forwarded to DihedralStyle.
            **kwargs: Keyword parameters forwarded to DihedralStyle.

        Returns:
            The registered (or existing) DihedralStyle instance.
        """
        return cast(DihedralStyle, self.def_style(DihedralStyle(name, *args, **kwargs)))

    def def_improperstyle(self, name: str, *args: Any, **kwargs: Any) -> ImproperStyle:
        """Create and register an ImproperStyle.

        Args:
            name: Name of the improper style (e.g. ``"cvff"``).
            *args: Positional parameters forwarded to ImproperStyle.
            **kwargs: Keyword parameters forwarded to ImproperStyle.

        Returns:
            The registered (or existing) ImproperStyle instance.
        """
        return cast(ImproperStyle, self.def_style(ImproperStyle(name, *args, **kwargs)))

    def def_pairstyle(self, name: str, *args: Any, **kwargs: Any) -> PairStyle:
        """Create and register a PairStyle.

        Args:
            name: Name of the pair style (e.g. ``"lj/cut"``).
            *args: Positional parameters forwarded to PairStyle.
            **kwargs: Keyword parameters forwarded to PairStyle.

        Returns:
            The registered (or existing) PairStyle instance.
        """
        return cast(PairStyle, self.def_style(PairStyle(name, *args, **kwargs)))

    def get_atomtypes(self) -> list[AtomType]:
        """Return all AtomType instances across all registered styles.

        Returns:
            Deduplicated list of AtomType instances.
        """
        return self.get_types(AtomType)

    def get_bondtypes(self) -> list[BondType]:
        """Return all BondType instances across all registered styles.

        Returns:
            Deduplicated list of BondType instances.
        """
        return self.get_types(BondType)

    def get_angletypes(self) -> list[AngleType]:
        """Return all AngleType instances across all registered styles.

        Returns:
            Deduplicated list of AngleType instances.
        """
        return self.get_types(AngleType)


# Import specialized Style and Type classes from potential.forcefield_styles
# No circular dependency since styles are now in potential, not core
def __getattr__(name: str):
    """Lazy import of specialized Style and Type classes."""
    specialized_names = {
        "AngleHarmonicStyle",
        "AngleHarmonicType",
        "BondHarmonicStyle",
        "BondHarmonicType",
        "DihedralOPLSStyle",
        "DihedralOPLSType",
        "PairCoulLongStyle",
        "PairLJ126CoulCutStyle",
        "PairLJ126CoulLongStyle",
        "PairLJ126Style",
        "PairLJ126Type",
    }
    if name in specialized_names:
        # Import from respective potential modules
        if name == "BondHarmonicStyle" or name == "BondHarmonicType":
            from molpy.potential.bond import BondHarmonicStyle, BondHarmonicType

            return (
                BondHarmonicStyle if name == "BondHarmonicStyle" else BondHarmonicType
            )
        elif name == "AngleHarmonicStyle" or name == "AngleHarmonicType":
            from molpy.potential.angle import AngleHarmonicStyle, AngleHarmonicType

            return (
                AngleHarmonicStyle
                if name == "AngleHarmonicStyle"
                else AngleHarmonicType
            )
        elif name == "DihedralOPLSStyle" or name == "DihedralOPLSType":
            from molpy.potential.dihedral import DihedralOPLSStyle, DihedralOPLSType

            return (
                DihedralOPLSStyle if name == "DihedralOPLSStyle" else DihedralOPLSType
            )
        elif name in [
            "PairCoulLongStyle",
            "PairLJ126CoulCutStyle",
            "PairLJ126CoulLongStyle",
            "PairLJ126Style",
            "PairLJ126Type",
        ]:
            from molpy.potential.pair import (
                PairCoulLongStyle,
                PairLJ126CoulCutStyle,
                PairLJ126CoulLongStyle,
                PairLJ126Style,
                PairLJ126Type,
            )

            return {
                "PairCoulLongStyle": PairCoulLongStyle,
                "PairLJ126CoulCutStyle": PairLJ126CoulCutStyle,
                "PairLJ126CoulLongStyle": PairLJ126CoulLongStyle,
                "PairLJ126Style": PairLJ126Style,
                "PairLJ126Type": PairLJ126Type,
            }[name]
        return getattr(styles, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
