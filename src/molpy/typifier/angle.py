"""Angle typifier — force-field agnostic."""

from collections import defaultdict

from molpy.core.atomistic import Angle
from molpy.core.forcefield import AngleType, AtomType, ForceField

from .base import TypifierBase


class AngleTypifier(TypifierBase):
    """Angle typifier.

    Matching rules:
    - "X" matches any atom type (wildcard)
    - Otherwise exact match by type or class
    - Bidirectional matching (A-B-C same as C-B-A)
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

        self._angle_table = {}
        for angle in self.ff.get_types(AngleType):
            self._angle_table[(angle.itom, angle.jtom, angle.ktom)] = angle

    def _matches(self, atomtype: AtomType, type_str: str) -> bool:
        at_type = atomtype.params.kwargs.get("type_", "X")
        at_class = atomtype.params.kwargs.get("class_", "X")

        if at_type == "X" or at_class == "X":
            return True

        return at_type == type_str or at_class == type_str

    def typify(self, elem: Angle) -> Angle:
        """Assign type to angle."""
        angle = elem
        itom_type = angle.itom.get("type", None)
        jtom_type = angle.jtom.get("type", None)
        ktom_type = angle.ktom.get("type", None)

        if None in (itom_type, jtom_type, ktom_type):
            if self.strict:
                raise ValueError(f"Angle atoms must have 'type' attribute: {angle}")
            return angle

        # Try direct type matching
        for (at1, at2, at3), angle_type in self._angle_table.items():
            if (
                self._matches(at1, itom_type)
                and self._matches(at2, jtom_type)
                and self._matches(at3, ktom_type)
            ) or (
                self._matches(at1, ktom_type)
                and self._matches(at2, jtom_type)
                and self._matches(at3, itom_type)
            ):
                angle.data["type"] = angle_type.name
                angle.data.update(**angle_type.params.kwargs)
                return angle

        # Try class matching
        itom_class = self._get_atom_class(itom_type)
        jtom_class = self._get_atom_class(jtom_type)
        ktom_class = self._get_atom_class(ktom_type)

        if itom_class and jtom_class and ktom_class:
            for (at1, at2, at3), angle_type in self._angle_table.items():
                if (
                    self._matches(at1, itom_class)
                    and self._matches(at2, jtom_class)
                    and self._matches(at3, ktom_class)
                ) or (
                    self._matches(at1, ktom_class)
                    and self._matches(at2, jtom_class)
                    and self._matches(at3, itom_class)
                ):
                    angle.data["type"] = angle_type.name
                    angle.data.update(**angle_type.params.kwargs)
                    return angle

        if not self.strict:
            return angle

        raise ValueError(
            f"No angle type found for atom types: {itom_type} - {jtom_type} - {ktom_type}"
        )

    def _get_atom_class(self, atom_type: str) -> str | None:
        if not hasattr(self, "_type_to_class"):
            self._type_to_class: dict[str, str] = {}
            for cls, types in self.class_to_types.items():
                for t in types:
                    self._type_to_class[t] = cls
        return self._type_to_class.get(atom_type)
