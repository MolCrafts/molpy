"""Unit tests for polymer Tool classes."""

import pytest

from molpy.tool.polymer import (
    BuildPolymer,
    BuildSystem,
    PlanSystem,
    PrepareMonomer,
)
from molpy.tool.base import Compute, Tool, ToolRegistry


# ---- Basic tests (no RDKit needed) ----


class TestPrepareMonomer:
    def test_is_tool_subclass(self):
        assert issubclass(PrepareMonomer, Tool)

    def test_not_compute_subclass(self):
        assert not issubclass(PrepareMonomer, Compute)

    def test_registered_in_registry(self):
        assert ToolRegistry.get("PrepareMonomer") is PrepareMonomer

    def test_frozen(self):
        tool = PrepareMonomer()
        with pytest.raises(AttributeError):
            tool.add_hydrogens = False

    def test_defaults(self):
        tool = PrepareMonomer()
        assert tool.add_hydrogens is True
        assert tool.optimize is True
        assert tool.gen_topology is True

    def test_callable(self):
        tool = PrepareMonomer()
        assert callable(tool)


class TestBuildPolymer:
    def test_is_tool_subclass(self):
        assert issubclass(BuildPolymer, Tool)

    def test_registered_in_registry(self):
        assert ToolRegistry.get("BuildPolymer") is BuildPolymer

    def test_frozen(self):
        tool = BuildPolymer()
        with pytest.raises(AttributeError):
            tool.reaction_preset = "other"

    def test_defaults(self):
        tool = BuildPolymer()
        assert tool.reaction_preset == "dehydration"
        assert tool.use_placer is True


class TestPlanSystem:
    def test_is_tool_subclass(self):
        assert issubclass(PlanSystem, Tool)

    def test_registered_in_registry(self):
        assert ToolRegistry.get("PlanSystem") is PlanSystem

    def test_defaults(self):
        tool = PlanSystem()
        assert tool.random_seed is None


class TestBuildSystem:
    def test_is_tool_subclass(self):
        assert issubclass(BuildSystem, Tool)

    def test_registered_in_registry(self):
        assert ToolRegistry.get("BuildSystem") is BuildSystem

    def test_defaults(self):
        tool = BuildSystem()
        assert tool.reaction_preset == "dehydration"
        assert tool.add_hydrogens is True
        assert tool.optimize is True
        assert tool.random_seed is None


# ---- PlanSystem functional test (no RDKit needed) ----


class TestPlanSystemRun:
    def test_plan_system_uniform(self):
        tool = PlanSystem(random_seed=42)
        result = tool.run(
            monomer_weights={"EO": 1.0},
            monomer_mass={"EO": 44.05},
            distribution_type="uniform",
            distribution_params={"p0": 5, "p1": 15},
            target_total_mass=5000.0,
        )
        assert "chains" in result
        assert "total_mass" in result
        assert "target_mass" in result
        assert len(result["chains"]) > 0
        assert result["target_mass"] == 5000.0
        # Total mass should be close to target
        assert result["total_mass"] > 0

    def test_plan_system_chain_structure(self):
        tool = PlanSystem(random_seed=0)
        result = tool.run(
            monomer_weights={"A": 0.7, "B": 0.3},
            monomer_mass={"A": 28.0, "B": 44.0},
            distribution_type="poisson",
            distribution_params={"p0": 10.0},
            target_total_mass=3000.0,
        )
        for chain in result["chains"]:
            assert "dp" in chain
            assert "monomers" in chain
            assert "mass" in chain
            assert chain["dp"] == len(chain["monomers"])
            assert chain["dp"] >= 1

    def test_plan_system_reproducible(self):
        tool = PlanSystem(random_seed=123)
        r1 = tool.run(
            monomer_weights={"X": 1.0},
            monomer_mass={"X": 50.0},
            distribution_type="uniform",
            distribution_params={"p0": 10, "p1": 20},
            target_total_mass=10000.0,
        )
        r2 = tool.run(
            monomer_weights={"X": 1.0},
            monomer_mass={"X": 50.0},
            distribution_type="uniform",
            distribution_params={"p0": 10, "p1": 20},
            target_total_mass=10000.0,
        )
        assert r1["chains"] == r2["chains"]


# ---- Functional tests requiring RDKit ----


@pytest.fixture
def eo_monomer():
    """Create an EO monomer for integration tests."""
    rdkit = pytest.importorskip("rdkit", reason="RDKit required")

    from molpy.adapter.rdkit import RDKitAdapter
    from molpy.tool.rdkit import Generate3D
    from molpy.parser.smiles import bigsmilesir_to_monomer, parse_bigsmiles

    ir = parse_bigsmiles("{[<]CCO[>]}")
    monomer = bigsmilesir_to_monomer(ir)
    adapter = RDKitAdapter(internal=monomer)
    gen3d = Generate3D(add_hydrogens=True, embed=True, optimize=False)
    adapter = gen3d(adapter)
    monomer = adapter.get_internal()
    monomer = monomer.get_topo(gen_angle=True, gen_dihe=True)
    for idx, atom in enumerate(monomer.atoms):
        atom["id"] = idx + 1
    return monomer


class TestPrepareMonomerRun:
    def test_run_simple_monomer(self):
        pytest.importorskip("rdkit", reason="RDKit required")
        tool = PrepareMonomer(optimize=False)
        monomer = tool.run("{[<]CCO[>]}")
        assert monomer is not None
        assert len(list(monomer.atoms)) > 0

    def test_run_without_3d(self):
        pytest.importorskip("rdkit", reason="RDKit required")
        tool = PrepareMonomer(
            add_hydrogens=False,
            optimize=False,
            gen_topology=False,
        )
        monomer = tool.run("{[<]CC[>]}")
        assert monomer is not None


class TestBuildPolymerRun:
    def test_run_linear_chain(self, eo_monomer):
        tool = BuildPolymer(reaction_preset="dehydration", use_placer=False)
        result = tool.run("{[#EO]|5}", {"EO": eo_monomer})
        assert "polymer" in result
        assert result["total_steps"] == 4  # 5 monomers, 4 connections
        assert len(result["connection_history"]) == 4
