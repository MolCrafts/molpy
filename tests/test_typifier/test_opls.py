"""
OPLS-AA typifier tests.

This module contains tests for OPLS atom type assignment and validation
against reference topology data.
"""

from pathlib import Path
from typing import List, Dict

import molpy as mp
import pytest


class TestOPLSTypifier:
    """Test suite for OPLS-AA force field typifier."""

    @pytest.fixture(autouse=True)
    def initdir(self, tmpdir):
        """Initialize temporary directory for tests."""
        tmpdir.chdir()

    @pytest.fixture
    def opls_validation_dir(self) -> Path:
        """Path to OPLS validation test data directory."""
        opls_dir = Path(__file__).parent.parent / "chemfile-testcases" / "forcefield" / "opls"
        if not opls_dir.exists():
            pytest.skip("OPLS validation data not available")
        return opls_dir

    @pytest.fixture
    def available_molecules(self, opls_validation_dir: Path) -> List[str]:
        """Get list of molecules available for testing."""
        molecules = []
        for molecule_dir in opls_validation_dir.iterdir():
            if molecule_dir.is_dir() and molecule_dir.name != "oplsaa.ff":
                gro_file = molecule_dir / f"{molecule_dir.name}.gro"
                top_file = molecule_dir / f"{molecule_dir.name}.top"
                if gro_file.exists() and top_file.exists():
                    molecules.append(molecule_dir.name)
        return sorted(molecules)

    @pytest.fixture
    def test_molecules(self, available_molecules: List[str]) -> List[str]:
        """Selected molecules for typifier testing."""
        # Choose a diverse set of molecules for testing
        priority_molecules = [
            "1-bromobutane", "acetone", "ethanol", "methane", "toluene", 
            "benzene", "propane", "methanol", "ethane"
        ]
        
        selected = []
        for mol in priority_molecules:
            if mol in available_molecules:
                selected.append(mol)
        
        # Add a few more if we don't have enough
        remaining = [m for m in available_molecules if m not in selected]
        selected.extend(remaining[:max(0, 10 - len(selected))])
        
        return selected[:10]

    @pytest.fixture
    def oplsaa_forcefield(self) -> mp.ForceField:
        """Load OPLS-AA forcefield."""
        # Use the built-in molpy function to load OPLS-AA
        try:
            frame = mp.io.read_xml_forcefield('oplsaa')
            return frame.forcefield
        except Exception:
            # Fallback: direct loading
            oplsaa_path = Path(mp.__file__).parent / 'data/forcefield/oplsaa.xml'
            from molpy.io.forcefield.xml import XMLForceFieldReader
            
            frame = mp.Frame()
            frame.forcefield = mp.ForceField()  # Initialize forcefield
            reader = XMLForceFieldReader(oplsaa_path)
            result = reader.read(frame)
            return frame.forcefield

    def get_molecule_files(self, molecule_name: str, opls_validation_dir: Path) -> tuple[Path, Path]:
        """Get GRO and TOP file paths for a molecule."""
        molecule_dir = opls_validation_dir / molecule_name
        gro_file = molecule_dir / f"{molecule_name}.gro"
        top_file = molecule_dir / f"{molecule_name}.top"
        return gro_file, top_file
    
    def load_reference_atom_types(self, molecule_name: str, opls_validation_dir: Path) -> List[str]:
        """Load reference atom types from topology file."""
        _, top_file = self.get_molecule_files(molecule_name, opls_validation_dir)
        
        ff = mp.ForceField()
        topology_ff = mp.io.read_top(str(top_file), ff)
        
        if topology_ff.n_atomstyles == 0:
            raise ValueError(f"No atom styles found in {molecule_name} topology")
        
        atomstyle = topology_ff.atomstyles[0]
        atomtypes = atomstyle.get_types()
        
        return [at.name for at in atomtypes]

    # Core typifier tests

    def test_oplsaa_forcefield_loading(self, oplsaa_forcefield: mp.ForceField):
        """Test that OPLS-AA forcefield loads with atom types."""
        assert oplsaa_forcefield is not None
        atomtypes = oplsaa_forcefield.get_atomtypes()
        assert len(atomtypes) > 0, "No atom types found in OPLS-AA forcefield"
        
        # Verify we have OPLS atom types
        opls_types = [at for at in atomtypes if at.name.startswith("opls_")]
        assert len(opls_types) > 100, f"Expected > 100 OPLS types, found {len(opls_types)}"
        print(f"✓ Loaded {len(opls_types)} OPLS atom types")

    def test_reference_atom_type_extraction(self, test_molecules: List[str], opls_validation_dir: Path):
        """Test extraction of reference atom types from topology files."""
        success_count = 0
        all_types = set()
        
        for molecule_name in test_molecules:
            try:
                reference_types = self.load_reference_atom_types(molecule_name, opls_validation_dir)
                assert len(reference_types) > 0, f"No atom types found for {molecule_name}"
                
                # Verify all are OPLS types
                non_opls = [t for t in reference_types if not t.startswith("opls_")]
                assert len(non_opls) == 0, f"Non-OPLS types in {molecule_name}: {non_opls}"
                
                all_types.update(reference_types)
                success_count += 1
                print(f"✓ {molecule_name}: {len(reference_types)} atoms, {len(set(reference_types))} unique types")
                
            except Exception as e:
                print(f"✗ {molecule_name}: {e}")
        
        assert success_count >= len(test_molecules) * 0.8, f"Too many failures: {success_count}/{len(test_molecules)}"
        print(f"✓ Found {len(all_types)} unique OPLS types across test molecules")

    @pytest.mark.skip(reason="Typifier implementation needed")
    def test_typifier_atom_type_assignment(self, test_molecules: List[str], oplsaa_forcefield: mp.ForceField, opls_validation_dir: Path):
        """Test typifier assignment against reference atom types."""
        # This is the main test - currently skipped until typifier is ready
        success_count = 0
        
        for molecule_name in test_molecules[:3]:  # Test subset first
            try:
                gro_file, top_file = self.get_molecule_files(molecule_name, opls_validation_dir)
                
                # Load structure
                frame = mp.Frame()
                structure = mp.io.read_gro(str(gro_file), frame)
                
                # Get reference atom types
                reference_types = self.load_reference_atom_types(molecule_name, opls_validation_dir)
                
                print(f"Testing {molecule_name}: {len(reference_types)} atoms")
                print(f"  Reference types: {set(reference_types)}")
                
                # TODO: Apply typifier when available
                # from molpy.typifier import OPLSTypifier
                # typifier = OPLSTypifier(oplsaa_forcefield)
                # typed_frame = typifier.assign_types(frame)
                # assigned_types = [atom.type for atom in typed_frame["atoms"]]
                
                # TODO: Compare assigned vs reference
                # assert len(assigned_types) == len(reference_types)
                # for i, (assigned, reference) in enumerate(zip(assigned_types, reference_types)):
                #     assert assigned == reference, f"Atom {i}: {assigned} != {reference}"
                
                success_count += 1
                
            except Exception as e:
                print(f"Failed {molecule_name}: {e}")
        
        print(f"Prepared typifier test for {success_count} molecules")

    def test_atom_type_coverage(self, test_molecules: List[str], oplsaa_forcefield: mp.ForceField, opls_validation_dir: Path):
        """Test that validation molecules cover diverse atom types."""
        validation_types = set()
        forcefield_types = set(at.name for at in oplsaa_forcefield.get_atomtypes() if at.name.startswith("opls_"))
        
        for molecule_name in test_molecules:
            try:
                reference_types = self.load_reference_atom_types(molecule_name, opls_validation_dir)
                validation_types.update(reference_types)
            except Exception:
                continue
        
        coverage = len(validation_types) / len(forcefield_types)
        print(f"Type coverage: {len(validation_types)}/{len(forcefield_types)} ({coverage:.1%})")
        
        # We should cover at least 20% of available OPLS types
        assert coverage >= 0.2, f"Poor type coverage: {coverage:.1%}"
        
        # Show most common types in validation set
        type_counts = {}
        for molecule_name in test_molecules:
            try:
                reference_types = self.load_reference_atom_types(molecule_name, opls_validation_dir)
                for atom_type in reference_types:
                    type_counts[atom_type] = type_counts.get(atom_type, 0) + 1
            except Exception:
                continue
        
        common_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("Most common atom types in validation set:")
        for atom_type, count in common_types:
            print(f"  {atom_type}: {count}")

    def test_charge_consistency(self, test_molecules: List[str], opls_validation_dir: Path):
        """Test that molecules are charge neutral."""
        neutral_count = 0
        
        for molecule_name in test_molecules:
            try:
                _, top_file = self.get_molecule_files(molecule_name, opls_validation_dir)
                
                ff = mp.ForceField()
                topology_ff = mp.io.read_top(str(top_file), ff)
                
                if topology_ff.n_atomstyles > 0:
                    atomstyle = topology_ff.atomstyles[0]
                    atomtypes = atomstyle.get_types()
                    
                    total_charge = sum(float(at.get("charge", 0)) for at in atomtypes)
                    is_neutral = abs(total_charge) < 1e-6
                    
                    if is_neutral:
                        neutral_count += 1
                        print(f"✓ {molecule_name}: neutral ({total_charge:.6f})")
                    else:
                        print(f"⚠ {molecule_name}: charge = {total_charge:.6f}")
                        
            except Exception as e:
                print(f"✗ {molecule_name}: {e}")
        
        # Most molecules should be neutral
        neutrality_rate = neutral_count / len(test_molecules)
        assert neutrality_rate >= 0.8, f"Too many non-neutral molecules: {neutrality_rate:.1%}"
        print(f"✓ Charge neutrality: {neutral_count}/{len(test_molecules)} ({neutrality_rate:.1%})")


# Utility functions for development

def debug_molecule_types(molecule_name: str, opls_validation_dir: Path) -> None:
    """Debug utility to analyze a specific molecule's atom types."""
    molecule_dir = opls_validation_dir / molecule_name
    gro_file = molecule_dir / f"{molecule_name}.gro"
    top_file = molecule_dir / f"{molecule_name}.top"
    
    if not gro_file.exists() or not top_file.exists():
        print(f"Missing files for {molecule_name}")
        return
    
    try:
        # Load structure
        frame = mp.Frame()
        structure = mp.io.read_gro(str(gro_file), frame)
        print(f"{molecule_name}: {structure['atoms'].nrows} atoms")
        
        # Load topology  
        ff = mp.ForceField()
        topology_ff = mp.io.read_top(str(top_file), ff)
        
        if topology_ff.n_atomstyles > 0:
            atomstyle = topology_ff.atomstyles[0]
            atomtypes = atomstyle.get_types()
            
            type_counts = {}
            for atomtype in atomtypes:
                type_name = atomtype.name
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            print("Atom types:")
            for atom_type, count in sorted(type_counts.items()):
                print(f"  {atom_type}: {count}")
                
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # For development and debugging
    opls_dir = Path(__file__).parent.parent / "chemfile-testcases" / "forcefield" / "opls"
    
    if opls_dir.exists():
        molecules = []
        for mol_dir in opls_dir.iterdir():
            if mol_dir.is_dir() and mol_dir.name != "oplsaa.ff":
                if (mol_dir / f"{mol_dir.name}.gro").exists() and (mol_dir / f"{mol_dir.name}.top").exists():
                    molecules.append(mol_dir.name)
        
        print(f"Found {len(molecules)} molecules in OPLS validation set")
        
        # Debug a few interesting molecules
        test_molecules = ["acetone", "benzene", "ethanol", "1-bromobutane"]
        for mol in test_molecules:
            if mol in molecules:
                debug_molecule_types(mol, opls_dir)
                print()
    else:
        print("OPLS validation directory not found")
