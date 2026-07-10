from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, override

from molpy.core import fields
from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from molpy.core.forcefield import (
    AngleType,
    AtomType,
    BondType,
    DihedralType,
    ForceField,
    ImproperType,
    PairType,
)

if TYPE_CHECKING:
    from molpy.core.affected_region import AffectedRegion
    from molpy.typifier.region import RegionTypes
    from molpy.typifier.scope import TypeScope


def _build_type_class_layer(
    ff: ForceField,
) -> tuple[dict[str, str], dict[str, int]]:
    """Map each atom-type *name* to its class and each class to its overlay layer.

    Bond/angle/dihedral types in OPLS-style force fields are stored with their
    class pair living only in the component *name* (e.g. ``"OW"``/``"HW"``); the
    atomtype objects carry no class. To match a bonded term we therefore resolve
    each atom's type to its class here and compare against those names.

    ``class_to_layer`` records the highest overlay layer of any atom type
    carrying a class, so a CL&P/CL&Pol class (layer ≥ 1) outranks a bare OPLS
    class (layer 0) when two parameter sets would otherwise tie.
    """
    type_to_class: dict[str, str] = {}
    class_to_layer: dict[str, int] = {}
    for at in ff.get_types(AtomType):
        at_type = at.params.kwargs.get("type_", "*")
        cls = at.params.kwargs.get("class_", "*")
        layer = int(at.params.kwargs.get("layer") or 0)
        # Only real atom types (those an atom is actually assigned) define the
        # type->class map. Skip the wildcard class-placeholder atomtypes the
        # reader creates for class-keyed bond endpoints (type_="*", name==class):
        # their name can collide with a real type (e.g. CL&P "FB" is both a BF4
        # fluorine type of class F AND the class of the NTf2/FSI fluorines).
        if at_type != "*":
            type_to_class[at.name] = cls
        if cls and cls != "*":
            class_to_layer[cls] = max(class_to_layer.get(cls, 0), layer)
    return type_to_class, class_to_layer


def _end_score(
    pattern_name: str | None, atom_type: str, atom_class: str | None
) -> int | None:
    """Specificity of one force-field type-component against one atom.

    Returns ``None`` if it does not match; otherwise higher is more specific:
    exact atom-type match (3) > class match (1) > wildcard ``*`` (0).
    """
    if pattern_name is None or pattern_name == "*":
        return 0
    if pattern_name == atom_type:
        return 3
    if atom_class is not None and pattern_name == atom_class:
        return 1
    return None


def _sequence_score(
    pattern_names: tuple[str, ...], atoms: list[tuple[str, str | None]]
) -> int | None:
    """Best specificity of a bonded-term pattern against an ordered atom list.

    Tries the term both forwards and reversed (bonded terms are symmetric under
    end-for-end reversal). ``atoms`` is ``[(type, class), ...]``. Returns the
    summed specificity of the best-matching orientation, or ``None``.
    """
    best: int | None = None
    for order in (atoms, atoms[::-1]):
        total = 0
        ok = True
        for pname, (at_type, at_class) in zip(pattern_names, order):
            s = _end_score(pname, at_type, at_class)
            if s is None:
                ok = False
                break
            total += s
        if ok and (best is None or total > best):
            best = total
    return best


def atomtype_matches(atomtype: AtomType, type_str: str) -> bool:
    """
    Check if an AtomType matches a given type string.

    Matching rules:
    1. If atomtype has a specific type (not "*"), compare by type
    2. If type doesn't match, compare by class

    Args:
        atomtype: AtomType instance
        type_str: Type string to match (from the atom's canonical type field or class name)

    Returns:
        True if matches, False otherwise
    """
    at_type = atomtype.params.kwargs.get("type_", "*")
    at_class = atomtype.params.kwargs.get("class_", "*")

    # Match by type first
    if at_type != "*" and at_type == type_str:
        return True

    # Then match by class
    if at_class != "*" and at_class == type_str:
        return True

    # If both are wildcards, also match
    return bool(at_type == "*" and at_class == "*")


class TypifierBase[T](ABC):
    def __init__(self, forcefield: ForceField) -> None:
        self.ff = forcefield

    @abstractmethod
    def typify(self, elem: T) -> T: ...


# ============================================================
# Generic force field typifiers (shared by OPLS, GAFF, etc.)
# ============================================================


class ForceFieldBondTypifier(TypifierBase[Bond]):
    """Match bond type based on atom types at both ends of the bond."""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield)
        self.strict = strict
        self._build_table()

    def _build_table(self):
        """Build the bond table plus type->class / class->layer maps."""
        self._type_to_class, self._class_to_layer = _build_type_class_layer(self.ff)
        self._bond_table = [
            ((bond.itom.name, bond.jtom.name), bond)
            for bond in self.ff.get_types(BondType)
        ]

    @override
    def typify(self, bond: Bond) -> Bond:
        """Assign the most specific (and highest-layer) matching bond type.

        Matches by resolving each atom's type to its class and comparing against
        the bond type's class pair (stored as the component *names*). Among all
        matches the most specific wins; ties break toward the higher overlay
        layer so CL&P/CL&Pol bonds override OPLS-AA.
        """
        itom_type = bond.itom.get("type", None)
        jtom_type = bond.jtom.get("type", None)
        if itom_type is None or jtom_type is None:
            raise ValueError(f"Bond atoms must have 'type' attribute: {bond}")

        atoms = [
            (itom_type, self._type_to_class.get(itom_type)),
            (jtom_type, self._type_to_class.get(jtom_type)),
        ]
        best_key: tuple[int, int] | None = None
        best_bt = None
        for (n1, n2), bond_type in self._bond_table:
            score = _sequence_score((n1, n2), atoms)
            if score is None:
                continue
            layer = max(
                self._class_to_layer.get(n1, 0), self._class_to_layer.get(n2, 0)
            )
            key = (score, layer)
            if best_key is None or key > best_key:
                best_key, best_bt = key, bond_type

        if best_bt is not None:
            bond.data[fields.TYPE.key] = best_bt.name
            bond.data.update(**best_bt.params.kwargs)
            return bond

        if not self.strict:
            return bond
        raise ValueError(
            f"No bond type found for atom types: {itom_type} - {jtom_type}"
        )


class ForceFieldAngleTypifier(TypifierBase[Angle]):
    """Match angle type based on atom types of three atoms in Angle"""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield)
        self.strict = strict
        self._build_table()

    def _build_table(self) -> None:
        """Build the angle table plus type->class / class->layer maps."""
        self._type_to_class, self._class_to_layer = _build_type_class_layer(self.ff)
        self._angle_table = [
            ((angle.itom.name, angle.jtom.name, angle.ktom.name), angle)
            for angle in self.ff.get_types(AngleType)
        ]

    @override
    def typify(self, angle: Angle) -> Angle:
        """Assign the most specific (highest-layer) matching angle type."""
        itom_type = angle.itom.get("type", None)
        jtom_type = angle.jtom.get("type", None)
        ktom_type = angle.ktom.get("type", None)

        if None in (itom_type, jtom_type, ktom_type):
            raise ValueError(f"Angle atoms must have 'type' attribute: {angle}")

        assert isinstance(itom_type, str)
        assert isinstance(jtom_type, str)
        assert isinstance(ktom_type, str)

        atoms = [
            (itom_type, self._type_to_class.get(itom_type)),
            (jtom_type, self._type_to_class.get(jtom_type)),
            (ktom_type, self._type_to_class.get(ktom_type)),
        ]
        best_key: tuple[int, int] | None = None
        best_at = None
        for names, angle_type in self._angle_table:
            score = _sequence_score(names, atoms)
            if score is None:
                continue
            layer = max(self._class_to_layer.get(n, 0) for n in names)
            key = (score, layer)
            if best_key is None or key > best_key:
                best_key, best_at = key, angle_type

        if best_at is not None:
            angle.data[fields.TYPE.key] = best_at.name
            angle.data.update(**best_at.params.kwargs)
            return angle

        if not self.strict:
            return angle
        raise ValueError(
            f"No angle type found for atom types: {itom_type} - {jtom_type} - {ktom_type}"
        )


class ForceFieldDihedralTypifier(TypifierBase[Dihedral]):
    """Match dihedral type based on atom types of four atoms in Dihedral"""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield)
        self.strict = strict
        self._build_table()

    def _build_table(self) -> None:
        """Build the dihedral table plus type->class / class->layer maps."""
        self._type_to_class, self._class_to_layer = _build_type_class_layer(self.ff)
        self._dihedral_table = [
            ((d.itom.name, d.jtom.name, d.ktom.name, d.ltom.name), d)
            for d in self.ff.get_types(DihedralType)
        ]

    @override
    def typify(self, dihedral: Dihedral) -> Dihedral:
        """Assign the most specific (highest-layer) matching dihedral type.

        OPLS dihedrals routinely use wildcard end atoms (``X-CT-CT-X``); the
        specificity score makes a fully-resolved pattern win over a partially
        wildcard one, and the layer tiebreak lets CL&P/CL&Pol override OPLS-AA.
        """
        itom_type = dihedral.itom.get("type", None)
        jtom_type = dihedral.jtom.get("type", None)
        ktom_type = dihedral.ktom.get("type", None)
        ltom_type = dihedral.ltom.get("type", None)

        if None in (itom_type, jtom_type, ktom_type, ltom_type):
            raise ValueError(f"Dihedral atoms must have 'type' attribute: {dihedral}")

        assert isinstance(itom_type, str)
        assert isinstance(jtom_type, str)
        assert isinstance(ktom_type, str)
        assert isinstance(ltom_type, str)

        atoms = [
            (itom_type, self._type_to_class.get(itom_type)),
            (jtom_type, self._type_to_class.get(jtom_type)),
            (ktom_type, self._type_to_class.get(ktom_type)),
            (ltom_type, self._type_to_class.get(ltom_type)),
        ]
        best_key: tuple[int, int] | None = None
        best_dt = None
        for names, dihedral_type in self._dihedral_table:
            score = _sequence_score(names, atoms)
            if score is None:
                continue
            layer = max(self._class_to_layer.get(n, 0) for n in names)
            key = (score, layer)
            if best_key is None or key > best_key:
                best_key, best_dt = key, dihedral_type

        if best_dt is not None:
            dihedral.data[fields.TYPE.key] = best_dt.name
            dihedral.data.update(**best_dt.params.kwargs)
            return dihedral

        if not self.strict:
            return dihedral
        raise ValueError(
            f"No dihedral type found for atom types: {itom_type} - {jtom_type} - {ktom_type} - {ltom_type}"
        )


class PairTypifier(TypifierBase[Atom]):
    """Assign nonbonded parameters (charge, sigma, epsilon) to atoms based on their types."""

    def __init__(self, forcefield: ForceField, strict: bool = True) -> None:
        super().__init__(forcefield)
        self.strict = strict
        self._build_pair_table()

    def _build_pair_table(self):
        """Build lookup table for pair types"""
        self._pair_table = {}
        for pair_type in self.ff.get_types(PairType):
            self._pair_table[pair_type.name] = pair_type

    @override
    def typify(self, atom: Atom) -> Atom:
        """Assign nonbonded parameters to atom based on its type"""
        atom_type = atom.get("type", None)

        if atom_type is None:
            if self.strict:
                raise ValueError(
                    f"Atom must have 'type' attribute before pair typification: {atom}"
                )
            return atom

        pair_type = self._pair_table.get(atom_type)

        if pair_type:
            atom.update(**pair_type.params.kwargs)
        elif self.strict:
            raise ValueError(f"No pair type found for atom type: {atom_type}")

        return atom


# ============================================================
# Generic atomistic typifier orchestrator
# ============================================================


class ForceFieldTypifier(TypifierBase[Atomistic]):
    """Base orchestrator that runs the full typing pipeline.

    Subclasses can override to use different atom typifiers or add
    additional typing steps (e.g., improper typing).
    """

    def __init__(
        self,
        forcefield: ForceField,
        skip_atom_typing: bool = False,
        skip_pair_typing: bool = False,
        skip_bond_typing: bool = False,
        skip_angle_typing: bool = False,
        skip_dihedral_typing: bool = False,
        strict_typing: bool = True,
    ) -> None:
        super().__init__(forcefield)
        self.skip_atom_typing = skip_atom_typing
        self.skip_pair_typing = skip_pair_typing
        self.skip_bond_typing = skip_bond_typing
        self.skip_angle_typing = skip_angle_typing
        self.skip_dihedral_typing = skip_dihedral_typing
        self.strict_typing = strict_typing

        self._init_typifiers()

    def _init_typifiers(self) -> None:
        """Initialize sub-typifiers. Subclasses override to customize."""
        if not self.skip_pair_typing:
            self.pair_typifier = PairTypifier(self.ff, strict=self.strict_typing)
        if not self.skip_bond_typing:
            self.bond_typifier = ForceFieldBondTypifier(
                self.ff, strict=self.strict_typing
            )
        if not self.skip_angle_typing:
            self.angle_typifier = ForceFieldAngleTypifier(
                self.ff, strict=self.strict_typing
            )
        if not self.skip_dihedral_typing:
            self.dihedral_typifier = ForceFieldDihedralTypifier(
                self.ff, strict=self.strict_typing
            )

    @property
    def scope(self) -> "TypeScope":
        """The receptive field of this typifier's atom typing, in bonds.

        ``reach = 2`` is the **measured** minimum: sweeping ``reach`` over
        1…5 and comparing region typing against whole-graph typing (the
        definitional oracle) for OPLS-AA, ``reach = 1`` mistypes 6 of 98 interior
        atoms of ``COCCOC`` while ``reach = 2`` reproduces every type on PEO,
        p-xylene (aromatic ring) and methyl acrylate (sp2 carbonyl). Anything
        larger only widens the extracted ball and fragments the retype cache.

        See :class:`~molpy.typifier.scope.TypeScope` for the two radii this
        implies (write-back ``ball(touched, 2)``, extraction ``ball(touched, 4)``).
        """
        from molpy.typifier.scope import TypeScope

        return TypeScope(reach=2)

    def _typify_relaxed(self, region: "AffectedRegion") -> Atomistic:
        """Type ``region`` with atom typing relaxed, then restore strictness.

        Atoms on the region's outer shell have truncated context and are *meant*
        to come back untyped; only the atom typifier's strictness would object.
        The interior guard in :meth:`RegionTypes.capture` is unconditional and
        does not depend on this flag — that is the whole point.
        """
        if self.skip_atom_typing:
            return self.typify(region)
        saved = self.atom_typifier.strict
        self.atom_typifier.strict = False
        try:
            return self.typify(region)
        finally:
            self.atom_typifier.strict = saved

    def typify_region(self, region: "AffectedRegion") -> "RegionTypes":
        """Type ``region`` as a standalone graph; snapshot its interior types."""
        from molpy.typifier.region import RegionTypes

        region_atoms = list(region.atoms)
        before = [dict(atom.data) for atom in region_atoms]
        typed = self._typify_relaxed(region)
        after = [dict(atom.data) for atom in typed.atoms]
        return RegionTypes.capture(region, typed, before, after)

    def retype_region(self, region: "AffectedRegion") -> "RegionTypes":
        """Type ``region`` and write its interior types onto the parent atoms.

        The un-cached one-shot used when no shared
        :class:`~molpy.typifier.cache.RetypeCache` is threaded through — types
        the region and applies the result via canonical order + ``entity_map``.
        """
        region_types = self.typify_region(region)
        region_types.apply_to(region)
        return region_types

    @override
    def typify(self, struct: Atomistic) -> Atomistic:
        """Return a new Atomistic with types assigned; input is not mutated."""
        if not self.skip_atom_typing:
            new_struct = self.atom_typifier.typify(struct)
        else:
            new_struct = struct.copy()

        if not self.skip_pair_typing:
            for atom in new_struct.atoms:
                if atom.get(fields.TYPE.key) is not None:
                    self.pair_typifier.typify(atom)

        if not self.skip_bond_typing:
            for bond in new_struct.bonds:
                if (
                    bond.itom.get(fields.TYPE.key) is not None
                    and bond.jtom.get(fields.TYPE.key) is not None
                ):
                    self.bond_typifier.typify(bond)

        if not self.skip_angle_typing:
            angles = new_struct.links.bucket(Angle)
            for angle in angles:
                endpoints = angle.endpoints
                if all(ep.get(fields.TYPE.key) is not None for ep in endpoints):
                    self.angle_typifier.typify(angle)

        if not self.skip_dihedral_typing:
            dihedrals = new_struct.links.bucket(Dihedral)
            for dihedral in dihedrals:
                endpoints = dihedral.endpoints
                if all(ep.get(fields.TYPE.key) is not None for ep in endpoints):
                    self.dihedral_typifier.typify(dihedral)

        return new_struct
