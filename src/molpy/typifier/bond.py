"""Bond typifier — force-field agnostic."""

from collections import defaultdict

from molpy.core.atomistic import Bond
from molpy.core.forcefield import AtomType, BondType, ForceField

from .base import TypifierBase


class BondTypifier(TypifierBase):
    """Bond typifier.

    Matching rules:
    - "X" matches any atom type (wildcard)
    - Otherwise exact match by type or class
    - Bidirectional matching (A-B same as B-A)
    """

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield, strict)
        self._build_tables()

    def _build_tables(self) -> None:
        """Build lookup tables for atom types and bonds."""
        self.class_to_types: dict[str, list[str]] = defaultdict(list)
        for at in self.ff.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "X")
            at_class = at.params.kwargs.get("class_", "X")
            if at_class != "X":
                if at_type != "X":
                    self.class_to_types[at_class].append(at_type)
                else:
                    self.class_to_types[at_class].append(at_class)

        self._bond_table = {}
        for bond in self.ff.get_types(BondType):
            self._bond_table[(bond.itom, bond.jtom)] = bond

    def _matches(self, atomtype: AtomType, type_str: str) -> bool:
        at_type = atomtype.params.kwargs.get("type_", "X")
        at_class = atomtype.params.kwargs.get("class_", "X")

        if at_type == "X" or at_class == "X":
            return True

        return at_type == type_str or at_class == type_str

    def typify(self, elem: Bond) -> Bond:
        """Assign type to bond."""
        bond = elem
        itom_type = bond.itom.get("type", None)
        jtom_type = bond.jtom.get("type", None)

        if itom_type is None or jtom_type is None:
            if self.strict:
                raise ValueError(f"Bond atoms must have 'type' attribute: {bond}")
            return bond

        # Try direct type matching
        for (at1, at2), bond_type in self._bond_table.items():
            if (self._matches(at1, itom_type) and self._matches(at2, jtom_type)) or (
                self._matches(at1, jtom_type) and self._matches(at2, itom_type)
            ):
                bond.data["type"] = bond_type.name
                bond.data.update(**bond_type.params.kwargs)
                return bond

        # Try class matching
        itom_class = self._get_atom_class(itom_type)
        jtom_class = self._get_atom_class(jtom_type)

        if itom_class and jtom_class:
            for (at1, at2), bond_type in self._bond_table.items():
                if (
                    self._matches(at1, itom_class) and self._matches(at2, jtom_class)
                ) or (
                    self._matches(at1, jtom_class) and self._matches(at2, itom_class)
                ):
                    bond.data["type"] = bond_type.name
                    bond.data.update(**bond_type.params.kwargs)
                    return bond

        if not self.strict:
            return bond

        raise ValueError(
            f"No bond type found for atom types: {itom_type} - {jtom_type}"
        )

    def _get_atom_class(self, atom_type: str) -> str | None:
        if not hasattr(self, "_type_to_class"):
            self._type_to_class: dict[str, str] = {}
            for cls, types in self.class_to_types.items():
                for t in types:
                    self._type_to_class[t] = cls
        return self._type_to_class.get(atom_type)
