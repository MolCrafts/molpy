"""Unit tests for :mod:`molpy.builder.polymer.ambertools.amber_builder`."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import molpy as mp
from molpy.builder.polymer.ambertools import AmberBuildResult, AmberPolymerBuilder
from molpy.builder.polymer.ambertools.amber_builder import _PreparedMonomer
from molpy.parser.smiles import parse_cgsmiles

CONDENSATION = mp.Reaction("[O;%a:1][H].[C:2][O;%b][H]>>[O:1][C:2]")


def _monomer() -> mp.Atomistic:
    monomer = mp.Atomistic()
    o1 = monomer.def_atom(element="O", name="O1", site="a")
    h1 = monomer.def_atom(element="H", name="H1")
    c1 = monomer.def_atom(element="C", name="C1")
    c2 = monomer.def_atom(element="C", name="C2")
    o2 = monomer.def_atom(element="O", name="O2", site="b")
    h2 = monomer.def_atom(element="H", name="H2")
    monomer.def_bonds([(o1, h1), (o1, c1), (c1, c2), (c2, o2), (o2, h2)])
    return monomer


class TestAmberPolymerBuilder:
    def test_constructor_uses_molpy_reaction_and_copies_templates(self):
        template = _monomer()
        builder = AmberPolymerBuilder({"M": template}, CONDENSATION)
        assert builder.reaction is CONDENSATION
        assert builder.library["M"] is not template
        assert builder.force_field == "gaff2"
        assert builder.charge_method == "bcc"
        assert builder.work_dir == Path("amber_work").resolve()

    def test_constructor_rejects_a_non_reaction(self):
        with pytest.raises(TypeError, match="molpy.Reaction"):
            AmberPolymerBuilder({"M": _monomer()}, "condensation")  # type: ignore[arg-type]

    def test_validation_rejects_unknown_labels_before_external_tools(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        with pytest.raises(ValueError, match="not found in library"):
            builder._validate_ir(parse_cgsmiles("{[#X]|2}"))

    def test_validation_rejects_non_linear_tleap_topology(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        ring = parse_cgsmiles("{[#M]1[#M][#M]1}")
        with pytest.raises(ValueError, match="one linear polymer path"):
            builder._validate_ir(ring)

    def test_compiles_standard_sites_and_reaction_into_amber_variants(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        ir = parse_cgsmiles("{[#M]|3}")
        builder._validate_ir(ir)
        recipes = builder._compile_semantics("{[#M]|3}", ir.base_graph)
        assert recipes == {
            "M": {
                "head": (None, "O1", ("H1",)),
                "chain": ("C2", "O1", ("H1", "H2", "O2")),
                "tail": ("C2", None, ("H2", "O2")),
            }
        }
        assert all(atom.get("port") is None for atom in builder.library["M"].atoms)

    def test_semantic_compilation_is_cached_by_cgsmiles(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        ir = parse_cgsmiles("{[#M]|3}")
        first = builder._compile_semantics("{[#M]|3}", ir.base_graph)
        second = builder._compile_semantics("{[#M]|3}", ir.base_graph)
        assert second is first

    def test_missing_standard_site_is_rejected_by_shared_polymer_builder(self):
        monomer = _monomer()
        for atom in monomer.atoms:
            atom["site"] = ""
        builder = AmberPolymerBuilder({"M": monomer}, CONDENSATION)
        ir = parse_cgsmiles("{[#M]|2}")
        with pytest.raises(ValueError, match="marks no reaction site"):
            builder._compile_semantics("{[#M]|2}", ir.base_graph)

    def test_build_writes_the_input_notation_to_the_result(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        result = AmberBuildResult(
            frame=MagicMock(),
            forcefield=MagicMock(),
            prmtop_path=Path("polymer.prmtop"),
            inpcrd_path=Path("polymer.inpcrd"),
            pdb_path=None,
            monomer_count=2,
            cgsmiles=None,
        )
        with (
            patch.object(builder, "_prepare_monomers"),
            patch.object(builder, "_build_with_tleap", return_value=result),
        ):
            built = builder.build("{[#M]|2}")
        assert built.cgsmiles == "{[#M]|2}"

    def test_sequence_uses_position_variants_not_monomer_ports(self):
        builder = AmberPolymerBuilder({"M": _monomer()}, CONDENSATION)
        builder._prepared_monomers["M"] = _PreparedMonomer(
            label="M",
            frcmod_file=Path("M.frcmod"),
            head_prepi=Path("HM.prepi"),
            chain_prepi=Path("M.prepi"),
            tail_prepi=Path("TM.prepi"),
            head_resname="HM",
            chain_resname="M",
            tail_resname="TM",
        )
        graph = parse_cgsmiles("{[#M]|5}").base_graph
        assert builder._build_sequence(graph) == "HM M M M TM"


class TestPreparedMonomer:
    def test_records_each_generated_residue_variant(self):
        prepared = _PreparedMonomer(
            label="M",
            frcmod_file=Path("M.frcmod"),
            head_prepi=Path("HM.prepi"),
            chain_prepi=None,
            tail_prepi=None,
            head_resname="HM",
            chain_resname=None,
            tail_resname=None,
        )
        assert prepared.label == "M"
        assert prepared.head_resname == "HM"
        assert prepared.chain_prepi is None
