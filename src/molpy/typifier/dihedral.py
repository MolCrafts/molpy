"""Dihedral typifier — force-field agnostic."""

from collections import defaultdict

from molpy.core.atomistic import Dihedral
from molpy.core.forcefield import AtomType, DihedralType, ForceField

from .base import TypifierBase


class DihedralTypifier(TypifierBase):
    """Dihedral typifier.

    Matching rules:
    - "X" matches any atom type (wildcard)
    - Otherwise exact match by type or class
    - Bidirectional matching (A-B-C-D same as D-C-B-A)
    """

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield, strict)
        self._build_tables()

    def _build_tables(self) -> None:
        self.class_to_types: dict[str, list[str]] = defaultdict(list)
        for at in self.ff.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "X")
            at_class = at.params.kwargs.get("class_", "X")
            if at_class != "X":
                if at_type != "X":
                    self.class_to_types[at_class].append(at_type)
                else:
                    self.class_to_types[at_class].append(at_class)

        self._dihedral_list: list[DihedralType] = list(self.ff.get_types(DihedralType))

    def _matches(self, atomtype: AtomType, type_str: str) -> bool:
        at_type = atomtype.params.kwargs.get("type_", "X")
        at_class = atomtype.params.kwargs.get("class_", "X")

        if at_type == "X" or at_class == "X":
            return True

        return at_type == type_str or at_class == type_str

    def typify(self, elem: Dihedral) -> Dihedral:
        """Assign type to dihedral."""
        dihedral = elem
        itom_type = dihedral.itom.get("type", None)
        jtom_type = dihedral.jtom.get("type", None)
        ktom_type = dihedral.ktom.get("type", None)
        ltom_type = dihedral.ltom.get("type", None)

        if None in (itom_type, jtom_type, ktom_type, ltom_type):
            if self.strict:
                raise ValueError(
                    f"Dihedral atoms must have 'type' attribute: {dihedral}"
                )
            return dihedral

        # Try direct type matching
        for dihedral_type in self._dihedral_list:
            at1, at2, at3, at4 = (
                dihedral_type.itom,
                dihedral_type.jtom,
                dihedral_type.ktom,
                dihedral_type.ltom,
            )
            if (
                self._matches(at1, itom_type)
                and self._matches(at2, jtom_type)
                and self._matches(at3, ktom_type)
                and self._matches(at4, ltom_type)
            ) or (
                self._matches(at1, ltom_type)
                and self._matches(at2, ktom_type)
                and self._matches(at3, jtom_type)
                and self._matches(at4, itom_type)
            ):
                dihedral.data["type"] = (
                    f"{itom_type}-{jtom_type}-{ktom_type}-{ltom_type}"
                )
                dihedral.data["ff_type"] = dihedral_type.name
                dihedral.data.update(**dihedral_type.params.kwargs)
                return dihedral

        # Try class matching
        itom_class = self._get_atom_class(itom_type)
        jtom_class = self._get_atom_class(jtom_type)
        ktom_class = self._get_atom_class(ktom_type)
        ltom_class = self._get_atom_class(ltom_type)

        if itom_class and jtom_class and ktom_class and ltom_class:
            for dihedral_type in self._dihedral_list:
                at1, at2, at3, at4 = (
                    dihedral_type.itom,
                    dihedral_type.jtom,
                    dihedral_type.ktom,
                    dihedral_type.ltom,
                )
                if (
                    self._matches(at1, itom_class)
                    and self._matches(at2, jtom_class)
                    and self._matches(at3, ktom_class)
                    and self._matches(at4, ltom_class)
                ) or (
                    self._matches(at1, ltom_class)
                    and self._matches(at2, ktom_class)
                    and self._matches(at3, jtom_class)
                    and self._matches(at4, itom_class)
                ):
                    dihedral.data["type"] = dihedral_type.name
                    dihedral.data.update(**dihedral_type.params.kwargs)
                    return dihedral

        if not self.strict:
            return dihedral

        raise ValueError(
            f"No dihedral type found for atom types: "
            f"{itom_type} - {jtom_type} - {ktom_type} - {ltom_type}"
        )

    def _get_atom_class(self, atom_type: str) -> str | None:
        if not hasattr(self, "_type_to_class"):
            self._type_to_class: dict[str, str] = {}
            for cls, types in self.class_to_types.items():
                for t in types:
                    self._type_to_class[t] = cls
        return self._type_to_class.get(atom_type)
