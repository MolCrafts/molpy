"""Unit tests for :mod:`molpy.builder.assembly._plan`."""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.builder.assembly._plan import (
    _AddedAtomRef,
    _AtomPatch,
    _CachedPatch,
    AssemblyCompiler,
    AssemblyPlan,
    LocalEnvironmentCache,
)
from molpy.core import fields
from molpy.typifier.base import Match, Typifier

NO_PLUS_O = "[N:1].[O:2]>>[N:1][O:2]"


class TestAddedAtomRef:
    def test_is_a_hashable_binding_and_creation_ordinal(self):
        reference = _AddedAtomRef(binding=2, ordinal=1)
        assert reference == _AddedAtomRef(2, 1)
        assert {reference: "created"}[_AddedAtomRef(2, 1)] == "created"


class TestAtomPatch:
    def test_preserves_atom_reference_and_sorted_annotations(self):
        patch = _AtomPatch(7, (("charge", 0.1), ("type", "CT")))
        assert patch.atom == 7
        assert dict(patch.annotations) == {"charge": 0.1, "type": "CT"}


class TestAssemblyPlan:
    def test_write_atoms_resolves_existing_and_reaction_created_handles(self):
        graph = mp.Atomistic()
        existing = graph.def_atom(element="N")
        created = graph.def_atom(element="O")
        plan = AssemblyPlan(
            bindings=({1: existing.handle},),
            atom_patches=(
                _AtomPatch(existing.handle, ((fields.TYPE.key, "N1"),)),
                _AtomPatch(
                    _AddedAtomRef(0, 0),
                    ((fields.TYPE.key, "O1"),),
                ),
            ),
        )

        plan.write_atoms(graph, [[created.handle]])

        assert existing[fields.TYPE] == "N1"
        assert created[fields.TYPE] == "O1"

    def test_missing_created_atom_is_a_contract_error(self):
        graph = mp.Atomistic()
        graph.def_atom(element="N")
        plan = AssemblyPlan((), (_AtomPatch(_AddedAtomRef(0, 0), ()),))
        with pytest.raises(RuntimeError, match="did not return it"):
            plan.write_atoms(graph, [])


class TestCachedPatch:
    def test_records_canonical_positions_not_world_handles(self):
        graph = mp.Atomistic()
        graph.def_atom(element="C")
        cached = _CachedPatch(graph, (0,), ((0, (("type", "CT"),)),))
        assert cached.root_positions == (0,)
        assert cached.atoms[0][0] == 0


class TestLocalEnvironmentCache:
    def test_isomorphic_rooted_products_reuse_typifier_output(self):
        calls = 0

        class CountingTypifier(Typifier):
            def match(self, graph) -> Match:
                nonlocal calls
                calls += 1
                return Match(
                    nodes=tuple(
                        {fields.TYPE.key: str(atom[fields.ELEMENT])}
                        for atom in graph.atoms
                    )
                )

        cache = LocalEnvironmentCache(CountingTypifier())
        first = self._bonded_pair("N", "O")
        second = self._bonded_pair("N", "O")

        first_patch = cache.patch(
            first,
            {first.atoms[0].handle},
            {atom.handle for atom in first.atoms},
        )
        second_patch = cache.patch(
            second,
            {second.atoms[0].handle},
            {atom.handle for atom in second.atoms},
        )

        assert calls == 1
        assert [values for _, values in first_patch] == [
            values for _, values in second_patch
        ]

    def test_different_context_shells_do_not_share_a_cache_entry(self):
        calls = 0

        class CountingTypifier(Typifier):
            def match(self, graph) -> Match:
                nonlocal calls
                calls += 1
                return Match(
                    nodes=tuple(
                        {fields.TYPE.key: str(atom[fields.ELEMENT])}
                        for atom in graph.atoms
                    )
                )

        cache = LocalEnvironmentCache(CountingTypifier())
        carbon = self._bonded_pair("C", "C")
        nitrogen_shell = self._bonded_pair("C", "N")
        cache.patch(carbon, {carbon.atoms[0].handle}, {carbon.atoms[0].handle})
        cache.patch(
            nitrogen_shell,
            {nitrogen_shell.atoms[0].handle},
            {nitrogen_shell.atoms[0].handle},
        )
        assert calls == 2

    def test_non_scalar_typifier_data_is_not_captured(self):
        assert LocalEnvironmentCache._scalar([1, 2]) is None
        assert LocalEnvironmentCache._scalar({"type": "CT"}) is None
        assert LocalEnvironmentCache._scalar("CT") == "CT"

    @staticmethod
    def _bonded_pair(left: str, right: str) -> mp.Atomistic:
        graph = mp.Atomistic()
        first = graph.def_atom(element=left)
        second = graph.def_atom(element=right)
        graph.def_bond(first, second)
        return graph


class TestAssemblyCompiler:
    def test_without_a_typifier_compiles_only_the_frozen_bindings(self):
        graph, binding = self._reactant_pair()
        plan = AssemblyCompiler(mp.Reaction(NO_PLUS_O), None, None).compile(
            graph, [binding], {}
        )
        assert plan.bindings == (binding,)
        assert plan.atom_patches == ()

    def test_compiles_then_replays_atom_types(self, element_typifier):
        reaction = mp.Reaction(NO_PLUS_O)
        graph, binding = self._reactant_pair()
        plan = AssemblyCompiler(reaction, element_typifier, 1).compile(
            graph, [binding], {}
        )

        _, created = reaction.apply_many_detailed(
            graph, list(plan.bindings), {}, refresh=False
        )
        plan.write_atoms(graph, created)

        assert {atom[fields.TYPE] for atom in graph.atoms} == {"t_N", "t_O"}

    def test_typifier_never_receives_the_assembled_world(self):
        seen: list[object] = []

        class SpyTypifier(Typifier):
            def match(self, graph) -> Match:
                seen.append(graph)
                return Match(
                    nodes=tuple(
                        {fields.TYPE.key: str(atom[fields.ELEMENT])}
                        for atom in graph.atoms
                    )
                )

        graph, binding = self._reactant_pair()
        AssemblyCompiler(mp.Reaction(NO_PLUS_O), SpyTypifier(), 1).compile(
            graph, [binding], {}
        )
        assert seen
        assert all(local is not graph for local in seen)

    def test_identical_motifs_are_typed_once(self):
        calls = 0

        class CountingTypifier(Typifier):
            def match(self, graph) -> Match:
                nonlocal calls
                calls += 1
                return Match(
                    nodes=tuple(
                        {fields.TYPE.key: str(atom[fields.ELEMENT])}
                        for atom in graph.atoms
                    )
                )

        graph = mp.Atomistic()
        bindings = []
        for _ in range(4):
            nitrogen = graph.def_atom(element="N")
            oxygen = graph.def_atom(element="O")
            bindings.append({1: nitrogen.handle, 2: oxygen.handle})

        AssemblyCompiler(mp.Reaction(NO_PLUS_O), CountingTypifier(), 1).compile(
            graph, bindings, {}
        )
        assert calls == 1

    def test_reaction_created_atoms_are_addressed_by_creation_ordinal(
        self, element_typifier
    ):
        reaction = mp.Reaction("[N:1].[C:2]>>[N:1][C:2]([O])[S]")
        graph = mp.Atomistic()
        nitrogen = graph.def_atom(element="N")
        carbon = graph.def_atom(element="C")
        binding = {1: nitrogen.handle, 2: carbon.handle}
        plan = AssemblyCompiler(reaction, element_typifier, 1).compile(
            graph, [binding], {}
        )
        _, created = reaction.apply_many_detailed(
            graph, list(plan.bindings), {}, refresh=False
        )
        plan.write_atoms(graph, created)
        added = [atom for atom in graph.atoms if atom[fields.ELEMENT] in {"O", "S"}]
        assert {(atom[fields.ELEMENT], atom[fields.TYPE]) for atom in added} == {
            ("O", "t_O"),
            ("S", "t_S"),
        }

    @staticmethod
    def _reactant_pair():
        graph = mp.Atomistic()
        nitrogen = graph.def_atom(element="N")
        oxygen = graph.def_atom(element="O")
        return graph, {1: nitrogen.handle, 2: oxygen.handle}
