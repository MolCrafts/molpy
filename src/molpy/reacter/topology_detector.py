"""
Topology detection and update for chemical reactions.

This module provides intelligent detection and updating of angles and dihedrals
after bond formation in reactions. It identifies affected old topology items
and replaces them with new ones based on the updated bond structure.
"""

from collections.abc import Collection

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from molpy.core.entity import Entity
from molpy.reacter.utils import find_neighbors


class TopologyDetector:
    """
    Detects and updates angles and dihedrals after reaction bond formation.

    This class implements intelligent topology detection that:
    1. Identifies atoms affected by new bonds
    2. Removes old topology items (angles/dihedrals) involving affected atoms
    3. Generates new topology items based on current bond structure
    """

    @staticmethod
    def _get_affected_atoms(
        assembly: Atomistic,
        new_bonds: Collection[Bond],
    ) -> set[Atom]:
        """
        Identify all atoms affected by new bond formation.

        Includes the endpoints of new bonds and all their neighbors.

        Args:
            assembly: The Atomistic structure
            new_bonds: List of newly formed bonds

        Returns:
            Set of affected atoms
        """
        affected = set()

        # Add all endpoints of new bonds
        for bond in new_bonds:
            affected.add(bond.itom)
            affected.add(bond.jtom)

            # Add all neighbors of endpoints
            neighbors_i = find_neighbors(assembly, bond.itom)
            neighbors_j = find_neighbors(assembly, bond.jtom)

            affected.update(neighbors_i)
            affected.update(neighbors_j)

        return affected

    @staticmethod
    def _remove_angles_involving_atoms(
        assembly: Atomistic,
        atoms: set[Atom],
    ) -> list[Angle]:
        """
        Remove all angles that involve any of the given atoms.

        Args:
            assembly: The Atomistic structure
            atoms: Set of atoms to check against

        Returns:
            List of removed angles
        """
        removed = []
        angles_to_remove = []

        # Find all angles involving affected atoms
        for angle in assembly.links.bucket(Angle):
            if angle.itom in atoms or angle.jtom in atoms or angle.ktom in atoms:
                angles_to_remove.append(angle)

        # Remove them from the structure
        if angles_to_remove:
            assembly.remove_link(*angles_to_remove)
            removed.extend(angles_to_remove)

        return removed

    @staticmethod
    def _remove_dihedrals_involving_atoms(
        assembly: Atomistic,
        atoms: set[Atom],
    ) -> list[Dihedral]:
        """
        Remove all dihedrals that involve any of the given atoms.

        Args:
            assembly: The Atomistic structure
            atoms: Set of atoms to check against

        Returns:
            List of removed dihedrals
        """
        removed = []
        dihedrals_to_remove = []

        # Find all dihedrals involving affected atoms
        for dihedral in assembly.links.bucket(Dihedral):
            if (
                dihedral.itom in atoms
                or dihedral.jtom in atoms
                or dihedral.ktom in atoms
                or dihedral.ltom in atoms
            ):
                dihedrals_to_remove.append(dihedral)

        # Remove them from the structure
        if dihedrals_to_remove:
            assembly.remove_link(*dihedrals_to_remove)
            removed.extend(dihedrals_to_remove)

        return removed

    @staticmethod
    def _remove_topology_with_removed_atoms(
        assembly: Atomistic,
        removed_atoms: Collection[Entity],
    ) -> tuple[list[Angle], list[Dihedral]]:
        """
        Remove all angles and dihedrals involving removed atoms.

        Args:
            assembly: The Atomistic structure
            removed_atoms: Collection of atoms that were removed

        Returns:
            Tuple of (removed_angles, removed_dihedrals)
        """
        removed_atoms_set = set(removed_atoms)
        removed_angles = []
        removed_dihedrals = []

        angles_to_remove = []
        for angle in assembly.links.bucket(Angle):
            if (
                angle.itom in removed_atoms_set
                or angle.jtom in removed_atoms_set
                or angle.ktom in removed_atoms_set
            ):
                angles_to_remove.append(angle)

        dihedrals_to_remove = []
        for dihedral in assembly.links.bucket(Dihedral):
            if (
                dihedral.itom in removed_atoms_set
                or dihedral.jtom in removed_atoms_set
                or dihedral.ktom in removed_atoms_set
                or dihedral.ltom in removed_atoms_set
            ):
                dihedrals_to_remove.append(dihedral)

        if angles_to_remove:
            assembly.remove_link(*angles_to_remove)
            removed_angles.extend(angles_to_remove)

        if dihedrals_to_remove:
            assembly.remove_link(*dihedrals_to_remove)
            removed_dihedrals.extend(dihedrals_to_remove)

        return removed_angles, removed_dihedrals

    @staticmethod
    def _generate_angles_around_bond(
        assembly: Atomistic,
        bond: Bond,
    ) -> list[Angle]:
        """
        Generate all angles that include the given bond as the middle edge.

        For bond (i, j), finds angles of form (k, i, j) and (i, j, k).

        Args:
            assembly: The Atomistic structure
            bond: The bond to generate angles around

        Returns:
            List of new Angle objects (not yet added to assembly)
        """
        new_angles = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors of atom_i (excluding atom_j)
        neighbors_i = find_neighbors(assembly, atom_i)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        # Find neighbors of atom_j (excluding atom_i)
        neighbors_j = find_neighbors(assembly, atom_j)
        neighbors_j = [n for n in neighbors_j if n is not atom_i]

        # Angles of form (k, i, j) where k is neighbor of i
        for k in neighbors_i:
            new_angles.append(Angle(k, atom_i, atom_j))

        # Angles of form (i, j, k) where k is neighbor of j
        for k in neighbors_j:
            new_angles.append(Angle(atom_i, atom_j, k))

        return new_angles

    @staticmethod
    def _generate_dihedrals_through_bond(
        assembly: Atomistic,
        bond: Bond,
    ) -> list[Dihedral]:
        """
        Generate all dihedrals that include the given bond.

        For bond (i, j), finds dihedrals of form (k, i, j, l) where
        k is neighbor of i and l is neighbor of j.

        Args:
            assembly: The Atomistic structure
            bond: The bond to generate dihedrals through

        Returns:
            List of new Dihedral objects (not yet added to assembly)
        """
        new_dihedrals = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors of atom_i (excluding atom_j)
        neighbors_i = find_neighbors(assembly, atom_i)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        # Find neighbors of atom_j (excluding atom_i)
        neighbors_j = find_neighbors(assembly, atom_j)
        neighbors_j = [n for n in neighbors_j if n is not atom_i]

        # Dihedrals of form (k, i, j, l)
        for k in neighbors_i:
            for l in neighbors_j:
                new_dihedrals.append(Dihedral(k, atom_i, atom_j, l))

        return new_dihedrals

    @staticmethod
    def _generate_dihedrals_continuing_from_bond(
        assembly: Atomistic,
        bond: Bond,
    ) -> list[Dihedral]:
        """
        Generate dihedrals that continue from the bond in both directions.

        For bond (i, j), generates:
        - Dihedrals extending forward: (i, j, k, l) where k is neighbor of j, l is neighbor of k
        - Dihedrals extending backward: (l, k, i, j) where k is neighbor of i, l is neighbor of k

        Args:
            assembly: The Atomistic structure
            bond: The bond to extend from

        Returns:
            List of new Dihedral objects
        """
        new_dihedrals = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors
        neighbors_i = find_neighbors(assembly, atom_i)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        neighbors_j = find_neighbors(assembly, atom_j)
        neighbors_j = [n for n in neighbors_j if n is not atom_i]

        # Dihedrals extending forward: (i, j, k, l)
        # where k is neighbor of j, l is neighbor of k
        for k in neighbors_j:
            neighbors_k = find_neighbors(assembly, k)
            neighbors_k = [n for n in neighbors_k if n not in (atom_i, atom_j)]
            for l in neighbors_k:
                new_dihedrals.append(Dihedral(atom_i, atom_j, k, l))

        # Dihedrals extending backward: (l, k, i, j)
        # where k is neighbor of i, l is neighbor of k
        for k in neighbors_i:
            neighbors_k = find_neighbors(assembly, k)
            neighbors_k = [n for n in neighbors_k if n not in (atom_i, atom_j)]
            for l in neighbors_k:
                new_dihedrals.append(Dihedral(l, k, atom_i, atom_j))

        return new_dihedrals

    @staticmethod
    def _deduplicate_angles(
        angles: list[Angle],
        existing_angles: Collection[Angle],
    ) -> list[Angle]:
        """
        Remove duplicate angles from the list.

        Two angles are considered duplicates if they involve the same three atoms
        (order-independent for endpoints, but order-dependent for middle atom).

        Args:
            angles: List of angles to deduplicate
            existing_angles: Existing angles in the assembly (to avoid duplicates)

        Returns:
            List of unique angles
        """
        # Create a set of existing angle tuples for fast lookup
        existing_tuples = set()
        for angle in existing_angles:
            # Normalize tuple: ensure consistent ordering (middle atom in center)
            atoms = (angle.itom, angle.jtom, angle.ktom)
            # Also check reverse: (ktom, jtom, itom)
            existing_tuples.add(atoms)
            existing_tuples.add((atoms[2], atoms[1], atoms[0]))

        unique_angles = []
        seen_tuples = set()

        for angle in angles:
            atoms = (angle.itom, angle.jtom, angle.ktom)
            # Check both forward and reverse
            if (
                atoms not in existing_tuples
                and (atoms[2], atoms[1], atoms[0]) not in existing_tuples
            ):
                if (
                    atoms not in seen_tuples
                    and (atoms[2], atoms[1], atoms[0]) not in seen_tuples
                ):
                    unique_angles.append(angle)
                    seen_tuples.add(atoms)
                    seen_tuples.add((atoms[2], atoms[1], atoms[0]))

        return unique_angles

    @staticmethod
    def _deduplicate_dihedrals(
        dihedrals: list[Dihedral],
        existing_dihedrals: Collection[Dihedral],
    ) -> list[Dihedral]:
        """
        Remove duplicate dihedrals from the list.

        Args:
            dihedrals: List of dihedrals to deduplicate
            existing_dihedrals: Existing dihedrals in the assembly

        Returns:
            List of unique dihedrals
        """
        # Create a set of existing dihedral tuples
        existing_tuples = set()
        for dihedral in existing_dihedrals:
            atoms = (dihedral.itom, dihedral.jtom, dihedral.ktom, dihedral.ltom)
            existing_tuples.add(atoms)
            # Dihedrals are directional, but reverse might be considered same
            existing_tuples.add((atoms[3], atoms[2], atoms[1], atoms[0]))

        unique_dihedrals = []
        seen_tuples = set()

        for dihedral in dihedrals:
            atoms = (dihedral.itom, dihedral.jtom, dihedral.ktom, dihedral.ltom)
            # Check both forward and reverse
            if (
                atoms not in existing_tuples
                and (atoms[3], atoms[2], atoms[1], atoms[0]) not in existing_tuples
            ):
                if (
                    atoms not in seen_tuples
                    and (atoms[3], atoms[2], atoms[1], atoms[0]) not in seen_tuples
                ):
                    unique_dihedrals.append(dihedral)
                    seen_tuples.add(atoms)
                    seen_tuples.add((atoms[3], atoms[2], atoms[1], atoms[0]))

        return unique_dihedrals

    @classmethod
    def detect_and_update_topology(
        cls,
        assembly: Atomistic,
        new_bonds: list[Bond],
        removed_atoms: Collection[Entity],
    ) -> tuple[list[Angle], list[Dihedral], list[Angle], list[Dihedral]]:
        """
        Detect and update topology structure after reaction.

        This method:
        1. Identifies atoms affected by new bonds
        2. Removes old topology items involving affected or removed atoms
        3. Generates new topology items based on current bond structure

        Args:
            assembly: The Atomistic structure (will be modified)
            new_bonds: List of newly formed bonds
            removed_atoms: List of atoms that were removed from the structure

        Returns:
            Tuple of (new_angles, new_dihedrals, removed_angles, removed_dihedrals)
        """
        # Step 1: Remove topology items involving removed atoms
        removed_angles_from_atoms, removed_dihedrals_from_atoms = (
            cls._remove_topology_with_removed_atoms(assembly, removed_atoms)
        )

        # Step 2: Identify affected atoms (endpoints of new bonds and their neighbors)
        affected_atoms = cls._get_affected_atoms(assembly, new_bonds)

        # Step 3: Remove old topology items involving affected atoms
        removed_angles_from_bonds = cls._remove_angles_involving_atoms(
            assembly, affected_atoms
        )
        removed_dihedrals_from_bonds = cls._remove_dihedrals_involving_atoms(
            assembly, affected_atoms
        )

        # Combine removed items
        all_removed_angles = removed_angles_from_atoms + removed_angles_from_bonds
        all_removed_dihedrals = (
            removed_dihedrals_from_atoms + removed_dihedrals_from_bonds
        )

        # Step 4: Generate new topology items based on new bonds
        # Get existing topology items for deduplication
        existing_angles = list(assembly.links.bucket(Angle))
        existing_dihedrals = list(assembly.links.bucket(Dihedral))

        new_angles_candidates = []
        new_dihedrals_candidates = []

        # Generate angles and dihedrals around each new bond
        for bond in new_bonds:
            # Angles around the bond
            angles_around_bond = cls._generate_angles_around_bond(assembly, bond)
            new_angles_candidates.extend(angles_around_bond)

            # Dihedrals through the bond
            dihedrals_through_bond = cls._generate_dihedrals_through_bond(
                assembly, bond
            )
            new_dihedrals_candidates.extend(dihedrals_through_bond)

            # Dihedrals continuing from the bond
            dihedrals_continuing = cls._generate_dihedrals_continuing_from_bond(
                assembly, bond
            )
            new_dihedrals_candidates.extend(dihedrals_continuing)

        # Step 5: Deduplicate and add new topology items
        unique_new_angles = cls._deduplicate_angles(
            new_angles_candidates, existing_angles
        )
        unique_new_dihedrals = cls._deduplicate_dihedrals(
            new_dihedrals_candidates, existing_dihedrals
        )

        # Add new items to assembly
        if unique_new_angles:
            assembly.add_link(*unique_new_angles)
        if unique_new_dihedrals:
            assembly.add_link(*unique_new_dihedrals)

        return (
            unique_new_angles,
            unique_new_dihedrals,
            all_removed_angles,
            all_removed_dihedrals,
        )
