from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, override

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

# Priority stride between force-field overlay layers. An overlay type (layer L)
# adds L * stride to its priority so it strictly outranks every lower-layer
# candidate in the SMARTS matcher regardless of specificity score. The stride
# is far larger than any realistic specificity score (pattern size) or
# overrides-based delta, so a single CL&P/CL&Pol type always wins over OPLS-AA
# where it matches, while OPLS-AA remains the fallback where it does not.
_LAYER_PRIORITY_STRIDE = 1000


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
        type_str: Type string to match (from Atom.data["type"] or class name)

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
            bond.data["type"] = best_bt.name
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
            angle.data["type"] = best_at.name
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
            dihedral.data["type"] = best_dt.name
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
# SMARTS-based atom typifier base class
# ============================================================


class ForceFieldAtomTypifier(TypifierBase["Atomistic"]):
    """Base class for SMARTS-based atom typifiers."""

    def __init__(
        self,
        forcefield: ForceField,
        strict: bool = False,
    ) -> None:
        super().__init__(forcefield)
        from .adapter import build_mol_graph

        self.pattern_dict = self._extract_patterns()
        self._build_mol_graph = build_mol_graph
        self.strict = strict

        from .layered_engine import LayeredTypingEngine

        self.engine = LayeredTypingEngine(self.pattern_dict)

    @abstractmethod
    def _extract_patterns(self) -> dict:
        """Extract SMARTS patterns from forcefield. Subclasses implement this."""
        ...

    @override
    def typify(self, struct: "Atomistic") -> "Atomistic":
        """Return a new Atomistic with atom types assigned; input is not mutated."""
        orig_atoms = list(struct.atoms)
        graph, vs_to_atomid, _atomid_to_vs = self._build_mol_graph(struct)

        result = self.engine.typify(graph, vs_to_atomid)

        new_struct = struct.copy()
        new_atoms = list(new_struct.atoms)

        for orig_atom, new_atom in zip(orig_atoms, new_atoms):
            atom_id = id(orig_atom)
            if atom_id in result:
                atomtype = result[atom_id]
                new_atom.data["type"] = atomtype

                atom_type_obj = self._find_atomtype_by_name(atomtype)
                if atom_type_obj:
                    # Don't overwrite existing atom fields with None kwargs from
                    # the AtomType (e.g. an AtomType with no `element` attr yields
                    # a kwargs dict carrying element=None, which would clobber the
                    # real element set during parsing).
                    new_atom.data.update(
                        {
                            k: v
                            for k, v in atom_type_obj.params.kwargs.items()
                            if v is not None
                        }
                    )

        if self.strict:
            untyped_atoms = [
                atom for atom in new_struct.atoms if atom.get("type") is None
            ]
            if untyped_atoms:
                untyped_info = [
                    f"{atom.get('element', '?')} (id={id(atom)})"
                    for atom in untyped_atoms[:10]
                ]
                error_msg = (
                    f"Failed to assign types to {len(untyped_atoms)} atom(s). "
                    f"Examples: {', '.join(untyped_info)}"
                )
                if len(untyped_atoms) > 10:
                    error_msg += f" (and {len(untyped_atoms) - 10} more)"
                raise ValueError(error_msg)

        return new_struct

    def _find_atomtype_by_name(self, name: str) -> AtomType | None:
        """Find AtomType object by name"""
        for at in self.ff.get_types(AtomType):
            if at.name == name:
                return at
        return None


# ============================================================
# OPLS-specific typifiers
# ============================================================


class _OplsAtomTypifier(ForceFieldAtomTypifier):
    """Assign atom types using SMARTS matcher for OPLS-AA force field.

    Internal helper for :class:`OplsTypifier`; not part of the public API.
    """

    def _extract_patterns(self):
        """Extract SMARTS patterns from OPLS forcefield with overrides-based priority."""
        from molpy.parser.smarts import SmartsParser

        from .graph import SMARTSGraph

        pattern_dict = {}
        atom_types = list(self.ff.get_types(AtomType))
        parser = SmartsParser()

        # Build overrides mapping
        overrides_map = {}
        for at in atom_types:
            overrides_str = at.params.kwargs.get("overrides")
            if overrides_str:
                overrides_map[at.name] = {s.strip() for s in overrides_str.split(",")}

        # Calculate priority based on overrides
        type_priority = {}
        for at in atom_types:
            explicit_priority = at.params.kwargs.get("priority")
            if explicit_priority is not None:
                try:
                    type_priority[at.name] = int(explicit_priority)
                    continue
                except (ValueError, TypeError):
                    pass

            priority = 0
            for _overrider, overridden_set in overrides_map.items():
                if at.name in overridden_set:
                    priority -= 1
            if at.name in overrides_map:
                priority += len(overrides_map[at.name])
            # Overlay layers (CL&P, CL&Pol read on top of OPLS-AA) outrank the
            # base force field: a type tagged layer=L beats every lower-layer
            # type regardless of specificity, while intra-layer overrides
            # ordering is preserved by adding (not replacing) the boost.
            layer = at.params.kwargs.get("layer")
            if layer:
                priority += int(layer) * _LAYER_PRIORITY_STRIDE
            type_priority[at.name] = priority

        for at in atom_types:
            smarts_str = at.params.kwargs.get("def_")

            if smarts_str:
                try:
                    priority = type_priority.get(at.name, 0)
                    overrides = overrides_map.get(at.name, set())

                    pattern = SMARTSGraph(
                        smarts_string=smarts_str,
                        parser=parser,
                        atomtype_name=at.name,
                        priority=priority,
                        source=f"oplsaa:{at.name}",
                        overrides=overrides,
                        target_vertices=[0],
                    )
                    pattern_dict[at.name] = pattern
                except Exception as e:
                    # A SMARTS pattern that fails to parse means this atom type
                    # would silently never match — a broken force-field
                    # definition. Fail fast instead of warning and dropping it.
                    raise ValueError(
                        f"Failed to parse SMARTS for atom type {at.name!r}: "
                        f"{smarts_str!r} ({e})"
                    ) from e

        return pattern_dict


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
    def context_radius(self) -> int:
        """Max SMARTS-pattern depth (in bonds) this typifier's atom typing needs.

        A conservative constant: OPLS/GAFF atom-typing SMARTS reach only a few
        bonds, so 3 covers their neighbour/ring context. Consumed by
        :func:`molpy.core.region_radius`, which floors it to the retype-safe 4 so
        an :class:`~molpy.core.AffectedRegion` always carries a complete shell.
        """
        return 3

    def typify_region(self, region: "AffectedRegion") -> "RegionTypes":
        """Type ``region`` as a standalone graph; snapshot its interior types.

        Thin delegate to :func:`molpy.typifier.region.typify_region`. Its
        presence is the capability marker the reacter checks (``hasattr``) before
        taking the region-scoped + cached retype path; typifiers without it fall
        back to the whole-graph pass unchanged.
        """
        from molpy.typifier.region import typify_region

        return typify_region(self, region)

    def retype_region(self, region: "AffectedRegion") -> "RegionTypes":
        """Type ``region`` and write its interior types onto the parent atoms.

        The un-cached one-shot the reacter uses when no shared
        :class:`~molpy.typifier.cache.RetypeCache` is threaded through — types
        the region and applies the result via canonical order + ``entity_map``.
        """
        from molpy.typifier.region import apply_region_types

        region_types = self.typify_region(region)
        apply_region_types(region_types, region)
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
                if atom.get("type") is not None:
                    self.pair_typifier.typify(atom)

        if not self.skip_bond_typing:
            for bond in new_struct.bonds:
                if (
                    bond.itom.get("type") is not None
                    and bond.jtom.get("type") is not None
                ):
                    self.bond_typifier.typify(bond)

        if not self.skip_angle_typing:
            angles = new_struct.links.bucket(Angle)
            for angle in angles:
                endpoints = angle.endpoints
                if all(ep.get("type") is not None for ep in endpoints):
                    self.angle_typifier.typify(angle)

        if not self.skip_dihedral_typing:
            dihedrals = new_struct.links.bucket(Dihedral)
            for dihedral in dihedrals:
                endpoints = dihedral.endpoints
                if all(ep.get("type") is not None for ep in endpoints):
                    self.dihedral_typifier.typify(dihedral)

        return new_struct


class OplsTypifier(ForceFieldTypifier):
    """OPLS-AA full typing orchestrator: atom → pair → bond → angle → dihedral."""

    def _init_typifiers(self) -> None:
        if not self.skip_atom_typing:
            self.atom_typifier = _OplsAtomTypifier(self.ff, strict=self.strict_typing)
        super()._init_typifiers()
