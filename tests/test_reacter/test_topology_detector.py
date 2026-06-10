#!/usr/bin/env python3
"""Unit tests for TopologyDetector class.

Tests cover:
- Topology preservation after bond formation (angles/dihedrals not affected should remain)
- New topology generation (angles/dihedrals around new bond)
- Removed topology tracking (only topology involving removed atoms)
"""

import pytest

from molpy.core.atomistic import Angle, Atom, Atomistic, Bond, Dihedral, Improper
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
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        # Original angles should still exist
        final_angles = len(list(struct.angles))
        assert final_angles >= initial_angles, (
            f"Angles were lost! Before: {initial_angles}, After: {final_angles}"
        )

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
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [], [d])

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
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

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
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        # Should generate dihedral: a-b-c-d
        assert len(new_dihedrals) >= 1, (
            f"Expected at least 1 new dihedral, got {len(new_dihedrals)}"
        )

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

        initial_angle_count = len(list(struct.angles))  # noqa: F841

        # Form a new bond (that doesn't create a-b-c angle since it already exists)
        new_bond = Bond(b, d)  # Creates b-c-d and a-b-d angles, not a-b-c
        struct.add_link(new_bond)

        # Run topology detector
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        # No duplicates should be created
        angle_tuples = set()
        for angle in struct.angles:
            tup = (angle.itom, angle.jtom, angle.ktom)
            rev_tup = (angle.ktom, angle.jtom, angle.itom)
            assert tup not in angle_tuples and rev_tup not in angle_tuples, (
                "Duplicate angle found!"
            )
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
        (
            new_angles,
            new_dihedrals,
            _new_impropers,
            removed_angles,
            removed_dihedrals,
            _removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        # All original angles should still exist
        final_angles = len(list(struct.angles))
        assert final_angles >= initial_angles, (
            f"Original angles lost! Before: {initial_angles}, After: {final_angles}"
        )

        # All original dihedrals should still exist
        final_dihedrals = len(list(struct.dihedrals))
        assert final_dihedrals >= initial_dihedrals, (
            f"Original dihedrals lost! Before: {initial_dihedrals}, After: {final_dihedrals}"
        )

        # New angle d-e-f should be added
        assert len(new_angles) >= 1


class TestImproperDetection:
    """Improper detection, deduplication, and removal (6-tuple API).

    Targets the planned API:

    - ``_generate_impropers_around_atoms(assembly, atoms)``
    - ``_deduplicate_impropers(candidates, existing)``
    - ``_remove_topology_with_removed_atoms`` returning a 3-tuple
    - ``detect_and_update_topology`` returning a 6-tuple
    """

    @staticmethod
    def _star(n_neighbors: int) -> tuple[Atomistic, Atom, list[Atom]]:
        """Build a star graph: one center bonded to ``n_neighbors`` atoms."""
        struct = Atomistic()
        center = Atom(symbol="C")
        neighbors = [Atom(symbol="H") for _ in range(n_neighbors)]
        struct.add_entity(center, *neighbors)
        struct.add_link(*[Bond(center, n) for n in neighbors])
        return struct, center, neighbors

    def test_improper_generated_for_three_neighbor_center(self) -> None:
        """An atom with exactly 3 bonded neighbors yields one Improper, center first."""
        struct, center, neighbors = self._star(3)

        impropers = TopologyDetector._generate_impropers_around_atoms(
            struct, [center, *neighbors]
        )

        assert len(impropers) == 1
        improper = impropers[0]
        # molpy convention: itom (first endpoint) is the central atom
        assert improper.itom is center
        assert {improper.jtom, improper.ktom, improper.ltom} == set(neighbors)

    def test_improper_not_generated_for_two_neighbor_atom(self) -> None:
        """An atom with only 2 bonded neighbors yields no improper."""
        struct, center, neighbors = self._star(2)

        impropers = TopologyDetector._generate_impropers_around_atoms(
            struct, [center, *neighbors]
        )

        assert impropers == []

    def test_improper_not_generated_for_four_neighbor_atom(self) -> None:
        """An atom with 4 bonded neighbors yields no improper (sp3, not sp2)."""
        struct, center, neighbors = self._star(4)

        impropers = TopologyDetector._generate_impropers_around_atoms(
            struct, [center, *neighbors]
        )

        assert impropers == []

    def test_improper_duplicate_candidate_dropped(self) -> None:
        """Candidate matching an existing improper (same center, unordered neighbors) is dropped."""
        struct, center, (n1, n2, n3) = self._star(3)
        existing = struct.def_improper(center, n1, n2, n3)

        # Same center, same neighbor set, permuted neighbor order
        candidate = Improper(center, n2, n1, n3)

        unique = TopologyDetector._deduplicate_impropers([candidate], [existing])

        assert unique == []

    def test_improper_non_duplicate_candidate_kept(self) -> None:
        """Candidate with a different center atom is not a duplicate."""
        struct, center, (n1, n2, n3) = self._star(3)
        existing = struct.def_improper(center, n1, n2, n3)

        # Different center -> different improper even with overlapping atoms
        candidate = Improper(n1, center, n2, n3)

        unique = TopologyDetector._deduplicate_impropers([candidate], [existing])

        assert unique == [candidate]

    def test_improper_removed_with_removed_atoms(self) -> None:
        """Impropers crossing removed atoms are removed; 3-tuple return."""
        struct, center, (n1, n2, n3) = self._star(3)
        struct.def_improper(center, n1, n2, n3)
        assert len(list(struct.impropers)) == 1

        removed_angles, removed_dihedrals, removed_impropers = (
            TopologyDetector._remove_topology_with_removed_atoms(struct, [n3])
        )

        assert removed_angles == []
        assert removed_dihedrals == []
        assert len(removed_impropers) == 1
        assert len(list(struct.impropers)) == 0

    def test_detect_and_update_topology_returns_six_tuple_with_impropers(self) -> None:
        """detect_and_update_topology returns the 6-tuple (incl. improper lists)."""
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(b, a), Bond(b, c))
        new_bond = Bond(b, d)
        struct.add_link(new_bond)

        result = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        assert len(result) == 6

    def test_detect_and_update_topology_adds_new_improper_to_assembly(self) -> None:
        """New bond making a 3-neighbor center generates the improper in-place."""
        struct = Atomistic()
        a = Atom(symbol="C")
        b = Atom(symbol="C")
        c = Atom(symbol="C")
        d = Atom(symbol="C")
        struct.add_entity(a, b, c, d)
        struct.add_link(Bond(b, a), Bond(b, c))
        new_bond = Bond(b, d)
        struct.add_link(new_bond)

        (
            new_angles,
            new_dihedrals,
            new_impropers,
            removed_angles,
            removed_dihedrals,
            removed_impropers,
        ) = TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        assert len(new_impropers) == 1
        improper = new_impropers[0]
        assert improper.itom is b
        assert {improper.jtom, improper.ktom, improper.ltom} == {a, c, d}
        # Added to the assembly, not just returned
        assert len(list(struct.impropers)) == 1
        assert removed_impropers == []


class TestAdjacencySingleBuild:
    """detect_and_update_topology builds the adjacency exactly once.

    Planned perf API (spec builder-reacter-05-perf): the detector calls
    ``build_adjacency`` once per invocation and threads it through every
    internal neighbor query — no full-bond-scan fallback calls.
    """

    @staticmethod
    def _ten_atom_structure_with_new_bond() -> tuple[Atomistic, Bond]:
        """Two 5-atom chains a0..a4 / b0..b4 joined by one new bond a4-b0."""
        struct = Atomistic()
        chain_a = [Atom(symbol="C") for _ in range(5)]
        chain_b = [Atom(symbol="C") for _ in range(5)]
        struct.add_entity(*chain_a, *chain_b)
        for i in range(4):
            struct.add_link(Bond(chain_a[i], chain_a[i + 1]))
            struct.add_link(Bond(chain_b[i], chain_b[i + 1]))
        new_bond = Bond(chain_a[4], chain_b[0])
        struct.add_link(new_bond)
        return struct, new_bond

    def test_adjacency_built_exactly_once_per_detect_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """build_adjacency is invoked exactly once per detector call."""
        import molpy.reacter.topology_detector as td_mod
        import molpy.reacter.utils as utils_mod

        struct, new_bond = self._ten_atom_structure_with_new_bond()

        build_calls = {"n": 0}
        real_build = utils_mod.build_adjacency

        def counting_build(assembly, *args, **kwargs):
            build_calls["n"] += 1
            return real_build(assembly, *args, **kwargs)

        monkeypatch.setattr(utils_mod, "build_adjacency", counting_build)
        if hasattr(td_mod, "build_adjacency"):
            monkeypatch.setattr(td_mod, "build_adjacency", counting_build)

        TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        assert build_calls["n"] == 1, (
            f"build_adjacency called {build_calls['n']} times; expected "
            f"exactly 1 per detect_and_update_topology invocation"
        )

    def test_adjacency_passed_to_all_queries_no_fallback_scans(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No find_neighbors call inside one detect runs in fallback mode."""
        import molpy.reacter.topology_detector as td_mod
        import molpy.reacter.utils as utils_mod

        struct, new_bond = self._ten_atom_structure_with_new_bond()

        fallback_calls = {"n": 0}
        real_find = utils_mod.find_neighbors

        def counting_find(assembly, atom, *args, **kwargs):
            if kwargs.get("adjacency") is None:
                fallback_calls["n"] += 1
            return real_find(assembly, atom, *args, **kwargs)

        # topology_detector imports find_neighbors by name; patch both
        # the source module attribute and the imported reference.
        monkeypatch.setattr(utils_mod, "find_neighbors", counting_find)
        monkeypatch.setattr(td_mod, "find_neighbors", counting_find)

        TopologyDetector.detect_and_update_topology(struct, [new_bond], [])

        assert fallback_calls["n"] == 0, (
            f"{fallback_calls['n']} find_neighbors calls ran in fallback "
            f"(adjacency=None) mode during one detect_and_update_topology "
            f"invocation; expected 0"
        )
