"""Unit tests for GBigSmilesCompiler."""

import pytest


from molpy.builder.polymer.compiler import CompilerConfig, GBigSmilesCompiler
from molpy.builder.polymer.system import Chain
from molpy.core.atomistic import Atomistic


class TestCompilerConfig:
    def test_defaults(self):
        config = CompilerConfig()
        assert config.reaction_preset == "dehydration"
        assert config.add_hydrogens is True
        assert config.optimize_geometry is True
        assert config.random_seed is None

    def test_frozen(self):
        config = CompilerConfig()
        with pytest.raises(AttributeError):
            config.reaction_preset = "other"

    def test_custom_values(self):
        config = CompilerConfig(
            reaction_preset="condensation",
            add_hydrogens=False,
            optimize_geometry=False,
            random_seed=42,
        )
        assert config.reaction_preset == "condensation"
        assert config.add_hydrogens is False
        assert config.random_seed == 42


class TestGBigSmilesCompiler:
    def test_init_default_config(self):
        compiler = GBigSmilesCompiler()
        assert compiler.config.reaction_preset == "dehydration"

    def test_init_custom_config(self):
        config = CompilerConfig(random_seed=99)
        compiler = GBigSmilesCompiler(config)
        assert compiler.config.random_seed == 99

    def test_chain_to_cgsmiles(self):
        chain = Chain(dp=3, monomers=["M0", "M1", "M0"], mass=100.0)
        result = GBigSmilesCompiler._chain_to_cgsmiles(chain)
        assert result == "{[#M0][#M1][#M0]}"

    def test_chain_to_cgsmiles_single(self):
        chain = Chain(dp=1, monomers=["A"], mass=50.0)
        result = GBigSmilesCompiler._chain_to_cgsmiles(chain)
        assert result == "{[#A]}"

    def test_estimate_monomer_mass(self):
        """Test mass estimation from atom symbols."""
        from molpy.core.atomistic import Atom

        monomer = Atomistic()
        monomer.add_entity(Atom(symbol="C"))
        monomer.add_entity(Atom(symbol="C"))
        monomer.add_entity(Atom(symbol="O"))

        compiler = GBigSmilesCompiler()
        masses = compiler._estimate_monomer_mass({"M0": monomer})
        # C=12.011 * 2 + O=15.999 = 40.021
        assert masses["M0"] == pytest.approx(40.021, abs=0.01)


class TestCompilerIntegration:
    """Integration tests requiring RDKit."""

    def test_compile_and_build_simple(self):
        """Test full pipeline with a simple G-BigSMILES string."""
        pytest.importorskip("rdkit", reason="RDKit required")
        from molpy.parser.smiles import parse_gbigsmiles

        # Simple: 5 repeat units, no distribution
        system_ir = parse_gbigsmiles("{[<]CCO[>]}|5|")
        config = CompilerConfig(
            optimize_geometry=False,
            random_seed=42,
        )
        compiler = GBigSmilesCompiler(config)
        chains = compiler.compile_and_build(system_ir)

        assert len(chains) >= 1
        assert isinstance(chains[0], Atomistic)
        assert len(list(chains[0].atoms)) > 0
