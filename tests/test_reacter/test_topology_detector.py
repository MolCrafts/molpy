#!/usr/bin/env python3
"""Unit tests for TopologyDetector class.

Tests cover:
- Topology preservation after bond formation (angles/dihedrals not affected should remain)
- New topology generation (angles/dihedrals around new bond)
- Removed topology tracking (only topology involving removed atoms)
"""

import pytest

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral
from molpy.reacter.topology_detector import TopologyDetector


class TestTopologyDetector:
    """Test TopologyDetector class."""

    def test_topology_preserved_after_bond_formation(self):
        """Test that angles/dihedrals not involving new bond are preserved."""
        # Create structure: A-B-C-D with A-B-C angle and B-C-D angle
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        e = Atom(symbol="C")  # New atom to form bond with
        struct.add_entity(a, b, c, d, e)
        struct.add_link(Bond(a, b), Bond(b, c), Bond(c, d))

        # Add angles that should be preserved
        angle_abc = Angle(a, b, c)
        angle_bcd = Angle(b, c, d)
        struct.add_link(angle_abc, angle_bcd)

        initial_angles = len(list(struct.angles))
        assert initial_angles == 2

        # Form new bond between d and e
        new_bond = Bond(d, e)
        struct.add_link(new_bond)

        # Run topology detector
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [new_bond], [])
        )

        # Original angles should still exist
        final_angles = len(list(struct.angles))
        assert (
            final_angles >= initial_angles
        ), f"Angles were lost! Before: {initial_angles}, After: {final_angles}"

        # New angles around the new bond should be added
        assert len(new_angles) >= 1, "No new angles generated around new bond"

        # No angles should be removed (no atoms removed)
        assert len(removed_angles) == 0

    def test_topology_removed_only_for_deleted_atoms(self):
        """Test that only topology involving removed atoms is deleted."""
        # Create structure: A-B-C-D with angles
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="H")  # This will be removed
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(a, b), Bond(b, c), Bond(c, d))

        # Add angles
        angle_abc = Angle(a, b, c)  # Should be preserved
        angle_bcd = Angle(b, c, d)  # Should be removed (involves d)
        struct.add_link(angle_abc, angle_bcd)

        initial_angles = len(list(struct.angles))
        assert initial_angles == 2

        # Remove atom d and its incident links
        struct.remove_entity(d, drop_incident_links=True)

        # Run topology detector with empty new bonds
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [], [d])
        )

        # angle_abc should still exist
        # angle_bcd was already removed by drop_incident_links
        remaining_angles = len(list(struct.angles))
        assert remaining_angles == 1, f"Expected 1 angle, got {remaining_angles}"

    def test_new_angles_generated_around_new_bond(self):
        """Test that new angles are generated around newly formed bond."""
        # Create structure: A-B and C-D (separate)
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(a, b), Bond(c, d))

        # No initial angles
        assert len(list(struct.angles)) == 0

        # Form new bond between b and c
        new_bond = Bond(b, c)
        struct.add_link(new_bond)

        # Run topology detector
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [new_bond], [])
        )

        # Should generate angles: a-b-c and b-c-d
        assert len(new_angles) == 2, f"Expected 2 new angles, got {len(new_angles)}"

        # Total angles in struct should be 2
        assert len(list(struct.angles)) == 2

    def test_new_dihedrals_generated_through_new_bond(self):
        """Test that new dihedrals are generated through newly formed bond."""
        # Create structure: A-B and C-D (separate)
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(a, b), Bond(c, d))

        # No initial dihedrals
        assert len(list(struct.dihedrals)) == 0

        # Form new bond between b and c
        new_bond = Bond(b, c)
        struct.add_link(new_bond)

        # Run topology detector
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [new_bond], [])
        )

        # Should generate dihedral: a-b-c-d
        assert (
            len(new_dihedrals) >= 1
        ), f"Expected at least 1 new dihedral, got {len(new_dihedrals)}"

    def test_no_duplicate_topology_items(self):
        """Test that duplicate angles/dihedrals are not added."""
        # Create structure with existing angle
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(a, b), Bond(b, c), Bond(c, d))

        # Add existing angle that would be generated
        existing_angle = Angle(a, b, c)
        struct.add_link(existing_angle)

        initial_angle_count = len(list(struct.angles))

        # Form a new bond (that doesn't create a-b-c angle since it already exists)
        new_bond = Bond(b, d)  # Creates b-c-d and a-b-d angles, not a-b-c
        struct.add_link(new_bond)

        # Run topology detector
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [new_bond], [])
        )

        # No duplicates should be created
        angle_tuples = set()
        for angle in struct.angles:
            tup = (angle.itom, angle.jtom, angle.ktom)
            rev_tup = (angle.ktom, angle.jtom, angle.itom)
            assert (
                tup not in angle_tuples and rev_tup not in angle_tuples
            ), f"Duplicate angle found!"
            angle_tuples.add(tup)

    def test_complex_reaction_topology_preservation(self):
        """Test that topology is preserved in a more complex reaction scenario."""
        # Create a chain: A-B-C-D-E with full topology
        struct = Atomistic()
        atoms = [Atom(symbol="C") for _ in range(5)]
        struct.add_atoms(atoms)
        a, b, c, d, e = atoms

        # Add bonds
        struct.add_link(Bond(a, b), Bond(b, c), Bond(c, d), Bond(d, e))

        # Add all angles
        struct.add_link(Angle(a, b, c), Angle(b, c, d), Angle(c, d, e))

        # Add dihedrals
        struct.add_link(Dihedral(a, b, c, d), Dihedral(b, c, d, e))

        initial_angles = len(list(struct.angles))
        initial_dihedrals = len(list(struct.dihedrals))

        # Create new atom and form bond with e
        f = Atom(symbol="C")
        struct.add_entity(f)
        new_bond = Bond(e, f)
        struct.add_link(new_bond)

        # Run topology detector (no atoms removed)
        new_angles, new_dihedrals, removed_angles, removed_dihedrals = (
            TopologyDetector.detect_and_update_topology(struct, [new_bond], [])
        )

        # All original angles should still exist
        final_angles = len(list(struct.angles))
        assert (
            final_angles >= initial_angles
        ), f"Original angles lost! Before: {initial_angles}, After: {final_angles}"

        # All original dihedrals should still exist
        final_dihedrals = len(list(struct.dihedrals))
        assert (
            final_dihedrals >= initial_dihedrals
        ), f"Original dihedrals lost! Before: {initial_dihedrals}, After: {final_dihedrals}"

        # New angle d-e-f should be added
        assert len(new_angles) >= 1
