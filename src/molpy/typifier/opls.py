"""OPLS-AA atom typifier and full typing orchestrator."""

import warnings

from molpy.core.atomistic import Angle, Atomistic, Dihedral
from molpy.core.forcefield import AtomType, ForceField

from .angle import AngleTypifier
from .base import TypifierBase
from .bond import BondTypifier
from .dihedral import DihedralTypifier
from .pair import PairTypifier


class OplsAtomTypifier(TypifierBase):
    """OPLS atom typifier using SMARTS patterns.

    OPLS uses:
    - "*" as wildcard in XML (converted to "X" internally)
    - Override-based priority system
    - Single SMARTS pattern per atom type
    """

    def __init__(
        self,
        forcefield: ForceField,
        strict: bool = False,
    ) -> None:
        super().__init__(forcefield, strict)
        from .adapter import build_mol_graph

        self.pattern_dict = self._extract_patterns()
        self._build_mol_graph = build_mol_graph

        from .layered_engine import LayeredTypingEngine

        self.engine = LayeredTypingEngine(self.pattern_dict)

    def _extract_patterns(self) -> dict:
        """Extract SMARTS patterns from OPLS forcefield."""
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
                    warnings.warn(
                        f"Failed to parse SMARTS for {at.name}: {smarts_str}, error: {e}",
                        stacklevel=2,
                    )

        return pattern_dict

    def typify(self, elem: Atomistic) -> Atomistic:
        """Assign types to all atoms in Atomistic structure.

        Returns a new Atomistic with typed atoms. The input is not modified.
        """
        struct = elem
        new_struct = struct.copy()
        graph, vs_to_atomid, _atomid_to_vs = self._build_mol_graph(new_struct)

        result = self.engine.typify(graph, vs_to_atomid)

        for atom in new_struct.atoms:
            atom_id = id(atom)
            if atom_id in result:
                atomtype = result[atom_id]
                atom.data["type"] = atomtype

                atom_type_obj = self._find_atomtype_by_name(atomtype)
                if atom_type_obj:
                    atom.data.update(**atom_type_obj.params.kwargs)

        if self.strict:
            untyped_atoms = [
                atom for atom in new_struct.atoms if atom.get("type") is None
            ]
            if untyped_atoms:
                MAX_ERROR_EXAMPLES = 10
                untyped_info = [
                    f"{atom.get('symbol', '?')} (id={id(atom)})"
                    for atom in untyped_atoms[:MAX_ERROR_EXAMPLES]
                ]
                error_msg = (
                    f"Failed to assign types to {len(untyped_atoms)} atom(s). "
                    f"Examples: {', '.join(untyped_info)}"
                )
                if len(untyped_atoms) > MAX_ERROR_EXAMPLES:
                    error_msg += (
                        f" (and {len(untyped_atoms) - MAX_ERROR_EXAMPLES} more)"
                    )
                raise ValueError(error_msg)

        return new_struct

    def _find_atomtype_by_name(self, name: str) -> AtomType | None:
        if not hasattr(self, "_atomtype_cache"):
            self._atomtype_cache: dict[str, AtomType] = {
                at.name: at for at in self.ff.get_types(AtomType)
            }
        return self._atomtype_cache.get(name)


class OplsTypifier(TypifierBase):
    """OPLS-AA full typing: atom -> pair -> bond -> angle -> dihedral."""

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
        super().__init__(forcefield, strict_typing)
        self.skip_atom_typing = skip_atom_typing
        self.skip_pair_typing = skip_pair_typing
        self.skip_bond_typing = skip_bond_typing
        self.skip_angle_typing = skip_angle_typing
        self.skip_dihedral_typing = skip_dihedral_typing

        if not self.skip_atom_typing:
            self.atom_typifier = OplsAtomTypifier(self.ff, strict=self.strict)
        if not self.skip_pair_typing:
            self.pair_typifier = PairTypifier(self.ff, strict=self.strict)
        if not self.skip_bond_typing:
            self.bond_typifier = BondTypifier(self.ff, strict=self.strict)
        if not self.skip_angle_typing:
            self.angle_typifier = AngleTypifier(self.ff, strict=self.strict)
        if not self.skip_dihedral_typing:
            self.dihedral_typifier = DihedralTypifier(self.ff, strict=self.strict)

    def typify(self, elem: Atomistic) -> Atomistic:
        """Run full typing pipeline.

        Returns a new Atomistic with all types assigned. Input is not modified.
        """
        struct = elem
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
