"""Unit tests for :mod:`molpy.builder._finalize`."""

import molpy as mp
from molpy.builder._finalize import Finalization, StructureFinalizer


def _linear_graph() -> mp.Atomistic:
    graph = mp.Atomistic()
    atoms = [graph.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(4)]
    for i in range(3):
        graph.def_bond(atoms[i], atoms[i + 1])
    graph.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)
    return graph


class TestFinalization:
    def test_stages_are_atoms_and_topology(self):
        assert [stage.value for stage in Finalization] == ["atoms", "topology"]


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
        graph.remove_link(*graph.angles, *graph.dihedrals)
        result = StructureFinalizer(Finalization.TOPOLOGY).apply(graph)
        first = (len(list(result.angles)), len(list(result.dihedrals)))
        StructureFinalizer(Finalization.TOPOLOGY).apply(result)
        assert first == (len(list(result.angles)), len(list(result.dihedrals)))
        assert first[0] > 0
