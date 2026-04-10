"""Tests for unified parser API (parse_smarts, parse_molecule, parse_mixture, parse_monomer, parse_polymer, top-level exports)."""

import molpy as mp
from molpy.parser import (
    parse_smarts,
    parse_molecule,
    parse_mixture,
    parse_monomer,
    parse_polymer,
    PolymerSpec,
)
from molpy.parser.smarts import SmartsParser
from molpy.core.atomistic import Atomistic


class TestParseSmartsFunction:
    """Test parse_smarts() free function matches SmartsParser class."""

    def test_simple_pattern(self):
        ir = parse_smarts("[C]")
        assert len(ir.atoms) == 1

    def test_two_atom_pattern(self):
        ir = parse_smarts("[C;X4][O;H1]")
        assert len(ir.atoms) == 2
        assert len(ir.bonds) == 1

    def test_matches_class_method(self):
        parser = SmartsParser()
        ir_class = parser.parse_smarts("[#6]")
        ir_func = parse_smarts("[#6]")
        assert len(ir_class.atoms) == len(ir_func.atoms)
        assert len(ir_class.bonds) == len(ir_func.bonds)


class TestParseMolecule:
    """Test parse_molecule() convenience wrapper."""

    def test_ethanol(self):
        mol = parse_molecule("CCO")
        assert isinstance(mol, Atomistic)
        assert len(mol.atoms) == 3
        assert len(mol.bonds) == 2

    def test_methane(self):
        mol = parse_molecule("C")
        assert isinstance(mol, Atomistic)
        assert len(mol.atoms) == 1

    def test_ethyl_acetate(self):
        mol = parse_molecule("CC(=O)OCC")
        assert isinstance(mol, Atomistic)
        assert len(mol.atoms) == 6  # 4 C + 2 O (SMILES only explicit heavy atoms)

    def test_benzene(self):
        mol = parse_molecule("c1ccccc1")
        assert isinstance(mol, Atomistic)
        assert len(mol.atoms) == 6


class TestParseMonomer:
    """Test parse_monomer() convenience wrapper."""

    def test_simple_monomer(self):
        monomer = parse_monomer("{[][<]CC[>][]}")
        assert isinstance(monomer, Atomistic)
        ports = [a for a in monomer.atoms if a.get("port")]
        assert len(ports) >= 2

    def test_polystyrene(self):
        monomer = parse_monomer("{[][<]CC(c1ccccc1)[>][]}")
        assert isinstance(monomer, Atomistic)
        assert len(monomer.atoms) > 2


class TestParseMixture:
    """Test parse_mixture() convenience wrapper."""

    def test_single_molecule(self):
        mols = parse_mixture("CCO")
        assert len(mols) == 1
        assert isinstance(mols[0], Atomistic)
        assert len(mols[0].atoms) == 3

    def test_dot_separated(self):
        mols = parse_mixture("C.CC")
        assert len(mols) == 2
        assert len(mols[0].atoms) == 1
        assert len(mols[1].atoms) == 2

    def test_salt_system(self):
        mols = parse_mixture("[Li+].[F-]")
        assert len(mols) == 2

    def test_always_returns_list(self):
        result = parse_mixture("C")
        assert isinstance(result, list)
        assert len(result) == 1


class TestParsePolymer:
    """Test parse_polymer() convenience wrapper."""

    def test_homopolymer(self):
        spec = parse_polymer("{[][<]CC[>][]}")
        assert isinstance(spec, PolymerSpec)
        assert spec.topology == "homopolymer"

    def test_copolymer(self):
        spec = parse_polymer("{[][<]CC[>],[<]CC(C)[>][]}")
        assert isinstance(spec, PolymerSpec)
        assert spec.topology == "random_copolymer"
        assert len(spec.all_monomers()) == 2

    def test_returns_polymerspec(self):
        spec = parse_polymer("{[][<]CC[>][]}")
        assert hasattr(spec, "segments")
        assert hasattr(spec, "topology")
        assert hasattr(spec, "all_monomers")


class TestNamespaceExports:
    """Test that all symbols exist in expected namespaces."""

    def test_parser_namespace(self):
        import molpy.parser as p

        assert hasattr(p, "parse_smiles")
        assert hasattr(p, "parse_bigsmiles")
        assert hasattr(p, "parse_gbigsmiles")
        assert hasattr(p, "parse_cgsmiles")
        assert hasattr(p, "parse_smarts")
        assert hasattr(p, "parse_molecule")
        assert hasattr(p, "parse_mixture")
        assert hasattr(p, "parse_monomer")
        assert hasattr(p, "parse_polymer")
        assert hasattr(p, "smilesir_to_atomistic")
        assert hasattr(p, "bigsmilesir_to_monomer")
        assert hasattr(p, "bigsmilesir_to_polymerspec")
        assert hasattr(p, "PolymerSpec")
        assert hasattr(p, "PolymerSegment")

    def test_submodule_access(self):
        """Parse functions live under mp.parser, not mp.*."""
        assert hasattr(mp.parser, "parse_molecule")
        assert hasattr(mp.parser, "parse_mixture")
        assert hasattr(mp.parser, "parse_monomer")
        assert hasattr(mp.parser, "parse_polymer")
        assert hasattr(mp.parser, "parse_smiles")
        assert hasattr(mp.parser, "parse_smarts")
        assert hasattr(mp.parser, "PolymerSpec")

    def test_tool_submodule_access(self):
        """generate_3d lives under mp.tool, not mp.*."""
        assert hasattr(mp.tool, "generate_3d")
        assert callable(mp.tool.generate_3d)

    def test_submodule_functions_work(self):
        mol = mp.parser.parse_molecule("CCO")
        assert isinstance(mol, Atomistic)
        assert len(mol.atoms) == 3

        mols = mp.parser.parse_mixture("C.CC")
        assert len(mols) == 2

        ir = mp.parser.parse_smarts("[C]")
        assert len(ir.atoms) == 1

        ir = mp.parser.parse_smiles("C")
        assert len(ir.atoms) == 1


class TestGenerateConvenience:
    """Test mp.tool.generate_3d() convenience function."""

    def test_generate_3d_returns_atomistic(self):
        pytest = __import__("pytest")
        try:
            from molpy.adapter import RDKitAdapter
        except ImportError:
            pytest.skip("RDKit not installed")
        if RDKitAdapter is None:
            pytest.skip("RDKit not installed")

        mol = mp.parser.parse_molecule("CCO")
        result = mp.tool.generate_3d(mol)
        assert isinstance(result, Atomistic)
        # Should have atoms with 3D coordinates
        assert len(result.atoms) >= len(mol.atoms)
        # Verify coordinates were generated
        first_atom = result.atoms[0]
        assert "x" in first_atom and "y" in first_atom and "z" in first_atom

    def test_generate_3d_no_hydrogens(self):
        pytest = __import__("pytest")
        try:
            from molpy.adapter import RDKitAdapter
        except ImportError:
            pytest.skip("RDKit not installed")
        if RDKitAdapter is None:
            pytest.skip("RDKit not installed")

        mol = mp.parser.parse_molecule("C")
        result = mp.tool.generate_3d(mol, add_hydrogens=False)
        assert isinstance(result, Atomistic)
