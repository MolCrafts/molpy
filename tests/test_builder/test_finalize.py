"""Unit tests for :mod:`molpy.builder._finalize`."""

import pytest

import molpy as mp
from molpy.builder._finalize import Finalization, StructureFinalizer
from molpy.core import fields
from molpy.typifier.forcefield import ForceFieldParams


def _linear_graph() -> mp.Atomistic:
    graph = mp.Atomistic()
    atoms = [graph.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(4)]
    graph.def_bonds([(atoms[i], atoms[i + 1]) for i in range(3)])
    graph.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
    return graph


class TestFinalization:
    def test_stages_are_stable_string_values(self):
        assert [stage.value for stage in Finalization] == [
            "atoms",
            "topology",
            "bonded",
        ]


class TestStructureFinalizer:
    def test_atoms_stage_removes_incomplete_higher_order_topology(self):
        graph = _linear_graph()
        result = StructureFinalizer(Finalization.ATOMS).apply(graph)
        assert result is graph
        assert list(result.bonds)
        assert not list(result.angles)
        assert not list(result.dihedrals)

    def test_topology_stage_materializes_complete_terms_once(self):
        graph = _linear_graph()
        graph.del_angle(*tuple(graph.angles))
        graph.del_dihedral(*tuple(graph.dihedrals))
        result = StructureFinalizer(Finalization.TOPOLOGY).apply(graph)
        first = (len(list(result.angles)), len(list(result.dihedrals)))
        StructureFinalizer(Finalization.TOPOLOGY).apply(result)
        assert first == (len(list(result.angles)), len(list(result.dihedrals)))
        assert first[0] > 0

    def test_bonded_stage_requires_a_parameter_assigner(self):
        with pytest.raises(TypeError, match="requires bonded"):
            StructureFinalizer(Finalization.BONDED)

    def test_parameter_assigner_is_rejected_for_other_stages(self):
        forcefield = mp.AtomisticForcefield()
        with pytest.raises(TypeError, match="only meaningful"):
            StructureFinalizer(
                Finalization.ATOMS,
                bonded=ForceFieldParams(forcefield),
            )

    def test_bonded_stage_assigns_relation_types(self):
        graph = _linear_graph()
        for atom in graph.atoms:
            atom[fields.TYPE] = "CT"

        forcefield = mp.AtomisticForcefield()
        atom_style = forcefield.def_atomstyle("full")
        ct = atom_style.def_type("CT", type_="CT", class_="CT")
        forcefield.def_bondstyle("harmonic").def_type(ct, ct, k=1.0, r0=1.5)
        forcefield.def_anglestyle("harmonic").def_type(ct, ct, ct, k=1.0, theta0=109.5)
        forcefield.def_dihedralstyle("opls").def_type(ct, ct, ct, ct, k1=1.0)

        result = StructureFinalizer(
            Finalization.BONDED,
            bonded=ForceFieldParams(forcefield),
        ).apply(graph)

        for collection in (result.bonds, result.angles, result.dihedrals):
            assert collection
            assert all(term.get(fields.TYPE) is not None for term in collection)
