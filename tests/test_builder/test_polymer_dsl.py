"""Unit tests for DSL entry functions."""

import pytest

from molpy.tool.polymer import (
    _detect_notation,
    polymer,
    polymer_system,
)
from molpy.core.atomistic import Atomistic


class TestDetectNotation:
    def test_gbigsmiles_pipe_annotation(self):
        assert _detect_notation("{[<]CCO[>]}|10|") == "gbigsmiles"

    def test_gbigsmiles_distribution(self):
        assert (
            _detect_notation("{[<]CCO[>]}|schulz_zimm(1500,3000)||5e5|") == "gbigsmiles"
        )

    def test_cgsmiles_fragments(self):
        assert _detect_notation("{[#EO]|10}.{#EO=[<]COC[>]}") == "cgsmiles_fragments"

    def test_pure_cgsmiles(self):
        assert _detect_notation("{[#EO]|10}") == "cgsmiles"

    def test_pure_cgsmiles_no_repeat(self):
        assert _detect_notation("{[#A][#B][#C]}") == "cgsmiles"


class TestPolymerPureCGSmiles:
    def test_requires_library(self):
        with pytest.raises(TypeError, match="requires 'library'"):
            polymer("{[#EO]|10}")

    def test_with_library(self):
        pytest.importorskip("rdkit", reason="RDKit required")

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

        result = polymer(
            "{[#EO]|3}",
            library={"EO": monomer},
            reaction_preset="dehydration",
            use_placer=False,
        )
        assert isinstance(result, Atomistic)
        assert len(list(result.atoms)) > 0


class TestPolymerGBigSMILES:
    def test_build_from_gbigsmiles(self):
        pytest.importorskip("rdkit", reason="RDKit required")

        result = polymer(
            "{[<]CCO[>]}|5|",
            optimize=False,
            random_seed=42,
        )
        assert isinstance(result, Atomistic)
        assert len(list(result.atoms)) > 0


class TestPolymerCGSmilesFragments:
    def test_build_with_fragments(self):
        pytest.importorskip("rdkit", reason="RDKit required")

        result = polymer(
            "{[#EO]|3}.{#EO=[<]CCO[>]}",
            optimize=False,
        )
        assert isinstance(result, Atomistic)
        assert len(list(result.atoms)) > 0


class TestPolymerSystem:
    def test_build_system(self):
        pytest.importorskip("rdkit", reason="RDKit required")

        chains = polymer_system(
            "{[<]CCO[>]}|5|",
            optimize=False,
            random_seed=42,
        )
        assert isinstance(chains, list)
        assert len(chains) >= 1
        for chain in chains:
            assert isinstance(chain, Atomistic)
