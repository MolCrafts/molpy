"""
Topology detection and update for chemical reactions.

This module provides intelligent detection and updating of angles, dihedrals,
and impropers after bond formation in reactions. It identifies affected old
topology items and replaces them with new ones based on the updated bond
structure.
"""

from collections.abc import Collection

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral, Improper
from molpy.core.entity import Entity
from molpy.reacter.utils import AdjacencyMap, build_adjacency, find_neighbors


class TopologyDetector:
    """Detects and updates bonded topology after reaction bond formation.

    Implements the topology bookkeeping required by REACTER-style template
    generation (Gissinger et al., Polymer 128 (2017) 211-217,
    DOI: 10.1016/j.polymer.2017.06.038; Gissinger et al., Macromolecules 53
    (2020) 9953-9961, DOI: 10.1021/acs.macromol.0c02012; LAMMPS
    ``fix bond/react``: https://docs.lammps.org/fix_bond_react.html):

    1. Identifies atoms affected by new bonds (endpoints + first shell)
    2. Removes topology items (angles/dihedrals/impropers) involving
       removed atoms
    3. Generates new angles/dihedrals around new bonds and improper
       candidates at affected sp2-like centers (exactly 3 bonded
       neighbors), deduplicated against existing topology

    Impropers matter physically: OPLS-AA and GAFF use them to keep sp2
    centers planar, so a post-reaction template that lost impropers would
    let planar groups pyramidalize.
    """

    @staticmethod
    def _generate_impropers_around_atoms(
        assembly: Atomistic,
        atoms: Collection[Atom],
        adjacency: AdjacencyMap | None = None,
    ) -> list[Improper]:
        """Generate improper candidates at atoms with exactly 3 neighbors.

        For each atom with exactly three bonded neighbors, emits
        ``Improper(center, n1, n2, n3)`` with the center atom in the i
        position (molpy convention, see :class:`molpy.core.atomistic.Improper`).
        Atoms with two or four+ neighbors yield no candidates.

        Args:
            assembly: The Atomistic structure.
            atoms: Candidate center atoms (typically the affected set).
            adjacency: Optional prebuilt adjacency map for O(degree) lookups.

        Returns:
            List of new Improper objects (not yet added to assembly).
        """
        candidates: list[Improper] = []
        for atom in atoms:
            neighbors = find_neighbors(assembly, atom, adjacency=adjacency)
            if len(neighbors) == 3:
                candidates.append(Improper(atom, *neighbors))
        return candidates

    @staticmethod
    def _deduplicate_impropers(
        impropers: list[Improper],
        existing_impropers: Collection[Improper],
    ) -> list[Improper]:
        """Remove duplicate impropers (same center, same unordered neighbors).

        Args:
            impropers: Candidate impropers.
            existing_impropers: Impropers already present in the assembly.

        Returns:
            List of unique new impropers.
        """

        def _key(improper: Improper) -> tuple[Atom, frozenset[Atom]]:
            endpoints = improper.endpoints
            return (endpoints[0], frozenset(endpoints[1:]))

        existing_keys = {_key(imp) for imp in existing_impropers}
        unique: list[Improper] = []
        seen: set[tuple[Atom, frozenset[Atom]]] = set()
        for improper in impropers:
            key = _key(improper)
            if key not in existing_keys and key not in seen:
                unique.append(improper)
                seen.add(key)
        return unique

    @staticmethod
    def _get_affected_atoms(
        assembly: Atomistic,
        new_bonds: Collection[Bond],
        adjacency: AdjacencyMap | None = None,
    ) -> set[Atom]:
        """
        Identify all atoms affected by new bond formation.

        Includes the endpoints of new bonds and all their neighbors.

        Args:
            assembly: The Atomistic structure
            new_bonds: List of newly formed bonds
            adjacency: Optional prebuilt adjacency map for O(degree) lookups.

        Returns:
            Set of affected atoms
        """
        affected = set()

        # Add all endpoints of new bonds
        for bond in new_bonds:
            affected.add(bond.itom)
            affected.add(bond.jtom)

            # Add all neighbors of endpoints
            neighbors_i = find_neighbors(assembly, bond.itom, adjacency=adjacency)
            neighbors_j = find_neighbors(assembly, bond.jtom, adjacency=adjacency)

            affected.update(neighbors_i)
            affected.update(neighbors_j)

        return affected

    @staticmethod
    def _remove_topology_with_removed_atoms(
        assembly: Atomistic,
        removed_atoms: Collection[Entity],
    ) -> tuple[list[Angle], list[Dihedral], list[Improper]]:
        """
        Remove all angles, dihedrals, and impropers involving removed atoms.

        Args:
            assembly: The Atomistic structure
            removed_atoms: Collection of atoms that were removed

        Returns:
            Tuple of (removed_angles, removed_dihedrals, removed_impropers)
        """
        removed_atoms_set = set(removed_atoms)
        removed_angles = []
        removed_dihedrals = []
        removed_impropers = []

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

        impropers_to_remove = []
        for improper in assembly.links.bucket(Improper):
            if any(ep in removed_atoms_set for ep in improper.endpoints):
                impropers_to_remove.append(improper)

        if angles_to_remove:
            assembly.remove_link(*angles_to_remove)
            removed_angles.extend(angles_to_remove)

        if dihedrals_to_remove:
            assembly.remove_link(*dihedrals_to_remove)
            removed_dihedrals.extend(dihedrals_to_remove)

        if impropers_to_remove:
            assembly.remove_link(*impropers_to_remove)
            removed_impropers.extend(impropers_to_remove)

        return removed_angles, removed_dihedrals, removed_impropers

    @staticmethod
    def _generate_angles_around_bond(
        assembly: Atomistic,
        bond: Bond,
        adjacency: AdjacencyMap | None = None,
    ) -> list[Angle]:
        """
        Generate all angles that include the given bond as the middle edge.

        For bond (i, j), finds angles of form (k, i, j) and (i, j, k).

        Args:
            assembly: The Atomistic structure
            bond: The bond to generate angles around
            adjacency: Optional prebuilt adjacency map for O(degree) lookups.

        Returns:
            List of new Angle objects (not yet added to assembly)
        """
        new_angles = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors of atom_i (excluding atom_j)
        neighbors_i = find_neighbors(assembly, atom_i, adjacency=adjacency)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        # Find neighbors of atom_j (excluding atom_i)
        neighbors_j = find_neighbors(assembly, atom_j, adjacency=adjacency)
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
        adjacency: AdjacencyMap | None = None,
    ) -> list[Dihedral]:
        """
        Generate all dihedrals that include the given bond.

        For bond (i, j), finds dihedrals of form (k, i, j, l) where
        k is neighbor of i and l is neighbor of j.

        Args:
            assembly: The Atomistic structure
            bond: The bond to generate dihedrals through
            adjacency: Optional prebuilt adjacency map for O(degree) lookups.

        Returns:
            List of new Dihedral objects (not yet added to assembly)
        """
        new_dihedrals = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors of atom_i (excluding atom_j)
        neighbors_i = find_neighbors(assembly, atom_i, adjacency=adjacency)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        # Find neighbors of atom_j (excluding atom_i)
        neighbors_j = find_neighbors(assembly, atom_j, adjacency=adjacency)
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
        adjacency: AdjacencyMap | None = None,
    ) -> list[Dihedral]:
        """
        Generate dihedrals that continue from the bond in both directions.

        For bond (i, j), generates:
        - Dihedrals extending forward: (i, j, k, l) where k is neighbor of j, l is neighbor of k
        - Dihedrals extending backward: (l, k, i, j) where k is neighbor of i, l is neighbor of k

        Args:
            assembly: The Atomistic structure
            bond: The bond to extend from
            adjacency: Optional prebuilt adjacency map for O(degree) lookups.

        Returns:
            List of new Dihedral objects
        """
        new_dihedrals = []
        atom_i = bond.itom
        atom_j = bond.jtom

        # Find neighbors
        neighbors_i = find_neighbors(assembly, atom_i, adjacency=adjacency)
        neighbors_i = [n for n in neighbors_i if n is not atom_j]

        neighbors_j = find_neighbors(assembly, atom_j, adjacency=adjacency)
        neighbors_j = [n for n in neighbors_j if n is not atom_i]

        # Dihedrals extending forward: (i, j, k, l)
        # where k is neighbor of j, l is neighbor of k
        for k in neighbors_j:
            neighbors_k = find_neighbors(assembly, k, adjacency=adjacency)
            neighbors_k = [n for n in neighbors_k if n not in (atom_i, atom_j)]
            for l in neighbors_k:
                new_dihedrals.append(Dihedral(atom_i, atom_j, k, l))

        # Dihedrals extending backward: (l, k, i, j)
        # where k is neighbor of i, l is neighbor of k
        for k in neighbors_i:
            neighbors_k = find_neighbors(assembly, k, adjacency=adjacency)
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
    ) -> tuple[
        list[Angle],
        list[Dihedral],
        list[Improper],
        list[Angle],
        list[Dihedral],
        list[Improper],
    ]:
        """
        Detect and update topology structure after reaction.

        This method:
        1. Removes topology items (angles/dihedrals/impropers) involving
           removed atoms
        2. Generates new angles/dihedrals around new bonds and improper
           candidates at affected atoms with exactly 3 bonded neighbors
        3. Adds new topology items (deduplicated with existing)

        Note: We do NOT remove topology involving "affected atoms" (bond endpoints
        and neighbors) because those angles/dihedrals don't need to change -
        they still exist with the same atoms. We only need to add NEW topology
        created by the new bond.

        Args:
            assembly: The Atomistic structure (will be modified)
            new_bonds: List of newly formed bonds
            removed_atoms: List of atoms that were removed from the structure

        Returns:
            Tuple of (new_angles, new_dihedrals, new_impropers,
            removed_angles, removed_dihedrals, removed_impropers)
        """
        # Step 1: Remove topology items involving removed atoms ONLY
        removed_angles, removed_dihedrals, removed_impropers = (
            cls._remove_topology_with_removed_atoms(assembly, removed_atoms)
        )

        # Step 2: Generate new topology items based on new bonds.
        # Build the adjacency map exactly once per call; every neighbor
        # query below is O(degree) against it (no full-bond scans).
        adjacency = build_adjacency(assembly)

        # Get existing topology items for deduplication
        existing_angles = list(assembly.links.bucket(Angle))
        existing_dihedrals = list(assembly.links.bucket(Dihedral))
        existing_impropers = list(assembly.links.bucket(Improper))

        new_angles_candidates = []
        new_dihedrals_candidates = []

        # Generate angles and dihedrals around each new bond
        for bond in new_bonds:
            # Angles around the bond
            angles_around_bond = cls._generate_angles_around_bond(
                assembly, bond, adjacency
            )
            new_angles_candidates.extend(angles_around_bond)

            # Dihedrals through the bond
            dihedrals_through_bond = cls._generate_dihedrals_through_bond(
                assembly, bond, adjacency
            )
            new_dihedrals_candidates.extend(dihedrals_through_bond)

            # Dihedrals continuing from the bond
            dihedrals_continuing = cls._generate_dihedrals_continuing_from_bond(
                assembly, bond, adjacency
            )
            new_dihedrals_candidates.extend(dihedrals_continuing)

        # Improper candidates at affected atoms (sp2-like: exactly 3 neighbors)
        affected_atoms = cls._get_affected_atoms(assembly, new_bonds, adjacency)
        new_impropers_candidates = cls._generate_impropers_around_atoms(
            assembly, affected_atoms, adjacency
        )

        # Step 3: Deduplicate and add new topology items
        unique_new_angles = cls._deduplicate_angles(
            new_angles_candidates, existing_angles
        )
        unique_new_dihedrals = cls._deduplicate_dihedrals(
            new_dihedrals_candidates, existing_dihedrals
        )
        unique_new_impropers = cls._deduplicate_impropers(
            new_impropers_candidates, existing_impropers
        )

        # Add new items to assembly
        if unique_new_angles:
            assembly.add_link(*unique_new_angles)
        if unique_new_dihedrals:
            assembly.add_link(*unique_new_dihedrals)
        if unique_new_impropers:
            assembly.add_link(*unique_new_impropers)

        return (
            unique_new_angles,
            unique_new_dihedrals,
            unique_new_impropers,
            removed_angles,
            removed_dihedrals,
            removed_impropers,
        )
