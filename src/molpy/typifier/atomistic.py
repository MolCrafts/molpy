# pyright: reportIncompatibleMethodOverride=false
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import override

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from molpy.core.forcefield import (
    AngleType,
    AtomType,
    BondType,
    DihedralType,
    ForceField,
    PairType,
)


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


class OplsBondTypifier(TypifierBase[Bond]):
    """Match bond type based on atom types at both ends of the bond.

    Strategy: Build class_to_types table and manually compare each AtomType
    """

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__(forcefield)
        self._build_table()

    def _build_table(self):
        """Build class_to_types table and bond table"""
        # Build class -> types mapping
        self.class_to_types: dict[str, list[str]] = defaultdict(list)
        for at in self.ff.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "*")
            at_class = at.params.kwargs.get("class_", "*")
            if at_class != "*":
                if at_type != "*":
                    self.class_to_types[at_class].append(at_type)
                else:
                    self.class_to_types[at_class].append(at_class)

        # Build bond table
        self._bond_table = {}
        for bond in self.ff.get_types(BondType):
            self._bond_table[(bond.itom, bond.jtom)] = bond

    @override
    def typify(self, bond: Bond) -> Bond:
        """Assign type to bond"""
        itom_type = bond.itom.get("type", None)
        jtom_type = bond.jtom.get("type", None)

        if itom_type is None or jtom_type is None:
            raise ValueError(f"Bond atoms must have 'type' attribute: {bond}")

        # Iterate through all bond types and match manually
        for (at1, at2), bond_type in self._bond_table.items():
            # Try forward and reverse matching
            if (
                atomtype_matches(at1, itom_type) and atomtype_matches(at2, jtom_type)
            ) or (
                atomtype_matches(at1, jtom_type) and atomtype_matches(at2, itom_type)
            ):
                bond.data["type"] = bond_type.name
                bond.data.update(**bond_type.params.kwargs)
                return bond

        # Not found, try class matching
        # Find class for itom_type and jtom_type
        # First get class from AtomType object
        itom_atomtype = None
        jtom_atomtype = None
        for at in self.ff.get_types(AtomType):
            if at.name == itom_type:
                itom_atomtype = at
            if at.name == jtom_type:
                jtom_atomtype = at

        itom_class = (
            itom_atomtype.params.kwargs.get("class_", "*") if itom_atomtype else None
        )
        jtom_class = (
            jtom_atomtype.params.kwargs.get("class_", "*") if jtom_atomtype else None
        )

        if itom_class and jtom_class and itom_class != "*" and jtom_class != "*":
            # Try matching class of AtomType objects in bond_type
            for (at1, at2), bond_type in self._bond_table.items():
                at1_class = (
                    at1.params.kwargs.get("class_", "*")
                    if hasattr(at1, "params")
                    else "*"
                )
                at2_class = (
                    at2.params.kwargs.get("class_", "*")
                    if hasattr(at2, "params")
                    else "*"
                )
                # Match class (support forward and reverse)
                if (at1_class == itom_class and at2_class == jtom_class) or (
                    at1_class == jtom_class and at2_class == itom_class
                ):
                    bond.data["type"] = bond_type.name
                    bond.data.update(**bond_type.params.kwargs)
                    return bond

        raise ValueError(
            f"No bond type found for atom types: {itom_type} - {jtom_type}"
        )


class OplsAngleTypifier(TypifierBase[Angle]):
    """Match angle type based on atom types of three atoms in Angle"""

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__(forcefield)
        self._build_table()

    def _build_table(self) -> None:
        """Build class_to_types table and angle table"""
        # Build class -> types mapping
        self.class_to_types: dict[str, list[str]] = defaultdict(list)
        for at in self.ff.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "*")
            at_class = at.params.kwargs.get("class_", "*")
            if at_class != "*":
                if at_type != "*":
                    self.class_to_types[at_class].append(at_type)
                else:
                    self.class_to_types[at_class].append(at_class)

        # Build angle table
        self._angle_table = {}
        for angle in self.ff.get_types(AngleType):
            self._angle_table[(angle.itom, angle.jtom, angle.ktom)] = angle

    @override
    def typify(self, angle: Angle) -> Angle:
        """Assign type to angle"""
        itom_type = angle.itom.get("type", None)
        jtom_type = angle.jtom.get("type", None)
        ktom_type = angle.ktom.get("type", None)

        if None in (itom_type, jtom_type, ktom_type):
            raise ValueError(f"Angle atoms must have 'type' attribute: {angle}")

        assert isinstance(itom_type, str)
        assert isinstance(jtom_type, str)
        assert isinstance(ktom_type, str)

        # Iterate through all angle types and match manually
        for (at1, at2, at3), angle_type in self._angle_table.items():
            # Try forward and reverse matching (center atom at2 unchanged)
            if (
                atomtype_matches(at1, itom_type)
                and atomtype_matches(at2, jtom_type)
                and atomtype_matches(at3, ktom_type)
            ) or (
                atomtype_matches(at1, ktom_type)
                and atomtype_matches(at2, jtom_type)
                and atomtype_matches(at3, itom_type)
            ):
                angle.data["type"] = angle_type.name
                angle.data.update(**angle_type.params.kwargs)
                return angle

        # Not found, try class matching
        itom_class = None
        jtom_class = None
        ktom_class = None
        for cls, types in self.class_to_types.items():
            if itom_type in types:
                itom_class = cls
            if jtom_type in types:
                jtom_class = cls
            if ktom_type in types:
                ktom_class = cls

        if itom_class and jtom_class and ktom_class:
            for (at1, at2, at3), angle_type in self._angle_table.items():
                if (
                    atomtype_matches(at1, itom_class)
                    and atomtype_matches(at2, jtom_class)
                    and atomtype_matches(at3, ktom_class)
                ) or (
                    atomtype_matches(at1, ktom_class)
                    and atomtype_matches(at2, jtom_class)
                    and atomtype_matches(at3, itom_class)
                ):
                    angle.data["type"] = angle_type.name
                    angle.data.update(**angle_type.params.kwargs)
                    return angle

        raise ValueError(
            f"No angle type found for atom types: {itom_type} - {jtom_type} - {ktom_type}"
        )


class OplsDihedralTypifier(TypifierBase[Dihedral]):
    """Match dihedral type based on atom types of four atoms in Dihedral"""

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__(forcefield)
        self._build_table()

    def _build_table(self) -> None:
        """Build class_to_types table and dihedral list"""
        # Build class -> types mapping
        self.class_to_types: dict[str, list[str]] = defaultdict(list)
        for at in self.ff.get_types(AtomType):
            at_type = at.params.kwargs.get("type_", "*")
            at_class = at.params.kwargs.get("class_", "*")
            if at_class != "*":
                if at_type != "*":
                    self.class_to_types[at_class].append(at_type)
                else:
                    self.class_to_types[at_class].append(at_class)

        # Build dihedral list (not dict! Multiple dihedrals may have same AtomType combination)
        self._dihedral_list: list[DihedralType] = list(self.ff.get_types(DihedralType))

    @override
    def typify(self, dihedral: Dihedral) -> Dihedral:
        """Assign type to dihedral"""
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

        # Iterate through all dihedral types and match manually
        for dihedral_type in self._dihedral_list:
            at1, at2, at3, at4 = (
                dihedral_type.itom,
                dihedral_type.jtom,
                dihedral_type.ktom,
                dihedral_type.ltom,
            )
            # Try forward and reverse matching
            if (
                atomtype_matches(at1, itom_type)
                and atomtype_matches(at2, jtom_type)
                and atomtype_matches(at3, ktom_type)
                and atomtype_matches(at4, ltom_type)
            ) or (
                atomtype_matches(at1, ltom_type)
                and atomtype_matches(at2, ktom_type)
                and atomtype_matches(at3, jtom_type)
                and atomtype_matches(at4, itom_type)
            ):
                dihedral.data["type"] = dihedral_type.name
                dihedral.data.update(**dihedral_type.params.kwargs)
                return dihedral

        # Not found, try class matching
        itom_class = None
        jtom_class = None
        ktom_class = None
        ltom_class = None
        for cls, types in self.class_to_types.items():
            if itom_type in types:
                itom_class = cls
            if jtom_type in types:
                jtom_class = cls
            if ktom_type in types:
                ktom_class = cls
            if ltom_type in types:
                ltom_class = cls

        if itom_class and jtom_class and ktom_class and ltom_class:
            for dihedral_type in self._dihedral_list:
                at1, at2, at3, at4 = (
                    dihedral_type.itom,
                    dihedral_type.jtom,
                    dihedral_type.ktom,
                    dihedral_type.ltom,
                )
                if (
                    atomtype_matches(at1, itom_class)
                    and atomtype_matches(at2, jtom_class)
                    and atomtype_matches(at3, ktom_class)
                    and atomtype_matches(at4, ltom_class)
                ) or (
                    atomtype_matches(at1, ltom_class)
                    and atomtype_matches(at2, ktom_class)
                    and atomtype_matches(at3, jtom_class)
                    and atomtype_matches(at4, itom_class)
                ):
                    dihedral.data["type"] = dihedral_type.name
                    dihedral.data.update(**dihedral_type.params.kwargs)
                    return dihedral

        raise ValueError(
            f"No dihedral type found for atom types: {itom_type} - {jtom_type} - {ktom_type} - {ltom_type}"
        )


class PairTypifier(TypifierBase[Atom]):
    """Assign nonbonded parameters (charge, sigma, epsilon) to atoms based on their types.

    This typifier reads PairType parameters from the forcefield and assigns them to atoms.
    """

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__(forcefield)
        self._build_pair_table()

    def _build_pair_table(self):
        """Build lookup table for pair types"""
        self._pair_table = {}
        print(self.ff.get_types(PairType))
        for pair_type in self.ff.get_types(PairType):
            self._pair_table[pair_type.name] = pair_type

    @override
    def typify(self, atom: Atom) -> Atom:
        """Assign nonbonded parameters to atom based on its type"""
        atom_type = atom.get("type", None)

        if atom_type is None:
            raise ValueError(
                f"Atom must have 'type' attribute before pair typification: {atom}"
            )

        # Find matching PairType
        pair_type = self._pair_table.get(atom_type)
        print(self._pair_table)

        if pair_type:
            # Assign charge, sigma, epsilon from PairType
            for key in ["charge", "sigma", "epsilon"]:
                if key in pair_type.params.kwargs:
                    atom.data[key] = pair_type.params.kwargs[key]
        else:
            raise ValueError(f"No pair type found for atom type: {atom_type}")

        return atom


class OplsAtomTypifier(TypifierBase["Atomistic"]):
    """Assign atom types using SMARTS matcher (support type references and dependency resolution)"""

    def __init__(self, forcefield: ForceField) -> None:
        super().__init__(forcefield)
        from .adapter import build_mol_graph

        # Extract patterns from forcefield
        self.pattern_dict = self._extract_patterns()
        self._build_mol_graph = build_mol_graph

        # Use LayeredTypingEngine
        from .layered_engine import LayeredTypingEngine

        self.engine = LayeredTypingEngine(self.pattern_dict)

    def _extract_patterns(self):
        """Extract or construct SMARTS patterns from forcefield

        Extract SMARTS definitions (def attribute) from OPLS forcefield AtomType and convert to SMARTSGraph objects.
        Support overrides attribute to control priority and type references (%opls_XXX).

        Returns:
            Dictionary mapping atom type name to SMARTSGraph
        """
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

        # Calculate priority: based on overrides and priority attributes
        type_priority = {}
        for at in atom_types:
            # First use explicit priority (if available)
            explicit_priority = at.params.kwargs.get("priority")
            if explicit_priority is not None:
                try:
                    type_priority[at.name] = int(explicit_priority)
                    continue
                except (ValueError, TypeError):
                    pass

            # Otherwise calculate based on overrides
            priority = 0
            # If this type is overridden by other types, lower priority
            for _overrider, overridden_set in overrides_map.items():
                if at.name in overridden_set:
                    priority -= 1
            # If this type overrides other types, increase priority
            if at.name in overrides_map:
                priority += len(overrides_map[at.name])
            type_priority[at.name] = priority

        # Extract SMARTS patterns
        for at in atom_types:
            smarts_str = at.params.kwargs.get("def_")

            if smarts_str:
                # Parse SMARTS string using SMARTSGraph
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
                        target_vertices=[0],  # Only type the first atom in the pattern
                    )
                    pattern_dict[at.name] = pattern
                except Exception as e:
                    # If parsing fails, log warning but continue
                    import warnings

                    warnings.warn(
                        f"Failed to parse SMARTS for {at.name}: {smarts_str}, error: {e}",
                        stacklevel=2,
                    )

        return pattern_dict

    @override
    def typify(self, struct: "Atomistic") -> "Atomistic":
        """Assign types to all atoms in Atomistic structure (using dependency-aware layered matching)"""
        # Convert molecule to graph
        graph, vs_to_atomid, _atomid_to_vs = self._build_mol_graph(struct)

        # Use LayeredTypingEngine for layered matching
        result = self.engine.typify(graph, vs_to_atomid)

        # Apply results to atoms
        for atom in struct.atoms:
            atom_id = id(atom)
            if atom_id in result:
                atomtype = result[atom_id]
                atom.data["type"] = atomtype

                # Get other parameters from forcefield AtomType
                atom_type_obj = self._find_atomtype_by_name(atomtype)
                if atom_type_obj:
                    atom.data.update(**atom_type_obj.params.kwargs)

        return struct

    def _find_atomtype_by_name(self, name: str) -> AtomType | None:
        """Find AtomType object by name"""
        for at in self.ff.get_types(AtomType):
            if at.name == name:
                return at
        return None


class OplsAtomisticTypifier(TypifierBase[Atomistic]):
    """Assign all types (bond, angle, dihedral) to entire Atomistic structure

    Note: This class assumes atoms are already assigned types. If you need to assign atom types simultaneously,
    use OplsAtomTypifier first, or use skip_atom_typing=False parameter.
    """

    def __init__(
        self,
        forcefield: ForceField,
        skip_atom_typing: bool = False,
        skip_pair_typing: bool = False,
        skip_bond_typing: bool = False,
        skip_angle_typing: bool = False,
        skip_dihedral_typing: bool = False,
    ) -> None:
        super().__init__(forcefield)
        self.skip_atom_typing = skip_atom_typing
        self.skip_pair_typing = skip_pair_typing
        self.skip_bond_typing = skip_bond_typing
        self.skip_angle_typing = skip_angle_typing
        self.skip_dihedral_typing = skip_dihedral_typing

        if not skip_atom_typing:
            self.atom_typifier = OplsAtomTypifier(forcefield)
        if not skip_pair_typing:
            self.pair_typifier = PairTypifier(forcefield)
        if not skip_bond_typing:
            self.bond_typifier = OplsBondTypifier(forcefield)
        if not skip_angle_typing:
            self.angle_typifier = OplsAngleTypifier(forcefield)
        if not skip_dihedral_typing:
            self.dihedral_typifier = OplsDihedralTypifier(forcefield)

    @override
    def typify(self, struct: Atomistic) -> Atomistic:
        """
        Assign types to all bonds, angles, dihedrals in Atomistic structure

        Args:
            struct: Atomistic structure

        Prerequisites:
            - If skip_atom_typing=True (default), all atoms must already have 'type' attribute
            - If skip_atom_typing=False, will assign atom types first
        """
        # Optional: First assign atom types
        if not self.skip_atom_typing:
            self.atom_typifier.typify(struct)

        # Assign pair types (nonbond parameters: charge, sigma, epsilon)
        if not self.skip_pair_typing:
            for atom in struct.atoms:
                self.pair_typifier.typify(atom)

        # Assign types to all bonds
        if not self.skip_bond_typing:
            for bond in struct.bonds:
                self.bond_typifier.typify(bond)

        # Assign types to all angles (if exist)
        if not self.skip_angle_typing:
            angles = struct.links.bucket(Angle)
            for angle in angles:
                self.angle_typifier.typify(angle)

        # Assign types to all dihedrals (if exist)
        if not self.skip_dihedral_typing:
            dihedrals = struct.links.bucket(Dihedral)
            for dihedral in dihedrals:
                self.dihedral_typifier.typify(dihedral)

        return struct
