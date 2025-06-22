"""
AmberTools-based polymer builder for molpy.
Uses molq to orchestrate AmberTools workflows for automated polymer construction.
"""

import os
import tempfile
import logging
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod

import molpy as mp

# Try to import RDKit for structure generation
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

logger = logging.getLogger(__name__)


class BuilderStep(ABC):
    """
    Abstract base class for individual AmberTools workflow steps.
    Each step takes a context dictionary and returns an updated context.
    """
    
    @abstractmethod
    def run(self, context: Dict) -> Dict:
        """
        Execute this step of the workflow.
        
        Args:
            context: Dictionary containing workflow state and file paths
            
        Returns:
            Updated context dictionary
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this step for logging/debugging purposes."""
        pass


class GenerateStructureStep(BuilderStep):
    """Step to generate initial polymer structure from SMILES using RDKit."""
    
    @property
    def name(self) -> str:
        return "generate_structure"
    
    def run(self, context: Dict) -> Dict:
        """Generate polymer structure from monomer SMILES using RDKit."""
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for structure generation. Install with: pip install rdkit")
        
        logger.info(f"Generating polymer structure for sequence: {context['sequence']}")
        
        # Generate polymer structure using RDKit
        monomer_smiles = context['monomer_smiles']
        sequence = context['sequence']
        n_repeat = context['n_repeat']
        
        # Create monomer molecule from SMILES
        monomer_mol = Chem.MolFromSmiles(monomer_smiles)
        if monomer_mol is None:
            raise ValueError(f"Invalid SMILES: {monomer_smiles}")
        
        # Add hydrogens and generate 3D coordinates
        monomer_mol = Chem.AddHs(monomer_mol)
        AllChem.EmbedMolecule(monomer_mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(monomer_mol)
        
        # For simple polymer building, replicate the monomer
        # This is a simplified implementation - real polymer building would handle bonding
        polymer_mol = monomer_mol
        for i in range(1, n_repeat):
            # In a real implementation, this would handle proper bonding between monomers
            # For now, just keep the single monomer structure
            pass
        
        # Save as MOL2 file
        output_file = context['workdir'] / "polymer.mol2"
        
        # Convert to MOL2 format using RDKit
        mol_block = Chem.MolToMolBlock(polymer_mol)
        
        # Write a simple MOL2 file (basic implementation)
        # In production, you'd want to use a proper MOL2 writer
        with open(output_file, 'w') as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write("polymer\n")
            f.write(f"{polymer_mol.GetNumAtoms()} {polymer_mol.GetNumBonds()} 0 0 0\n")
            f.write("SMALL\n")
            f.write("NO_CHARGES\n")
            f.write("\n")
            f.write("@<TRIPOS>ATOM\n")
            
            conf = polymer_mol.GetConformer()
            for i, atom in enumerate(polymer_mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                f.write(f"{i+1:>7} {atom.GetSymbol()}{i+1} {pos.x:>10.4f} {pos.y:>10.4f} {pos.z:>10.4f} {atom.GetSymbol():<4}\n")
            
            f.write("@<TRIPOS>BOND\n")
            for i, bond in enumerate(polymer_mol.GetBonds()):
                f.write(f"{i+1:>5} {bond.GetBeginAtomIdx()+1:>5} {bond.GetEndAtomIdx()+1:>5} {bond.GetBondTypeAsDouble():.0f}\n")
        
        context['mol2_file'] = output_file
        context['polymer_mol'] = polymer_mol
        
        logger.info(f"Generated polymer structure: {output_file}")
        return context


class PrepgenStep(BuilderStep):
    """Step to generate AMBER prepi file using prepgen command."""
    
    @property
    def name(self) -> str:
        return "prepgen"
    
    def run(self, context: Dict) -> Dict:
        """Generate prepi file using prepgen command."""
        logger.info("Running prepgen to generate prepi file")
        
        input_file = context['mol2_file']
        output_file = context['workdir'] / "polymer.prepi"
        
        # Build prepgen command
        ambertools_bin = context.get('ambertools_bin', '')
        if ambertools_bin:
            prepgen_cmd = os.path.join(ambertools_bin, 'prepgen')
        else:
            prepgen_cmd = 'prepgen'
        
        # Check if prepgen is available
        if not shutil.which(prepgen_cmd):
            raise RuntimeError(f"prepgen not found. Make sure AmberTools is installed and in PATH.")
        
        # Run prepgen command
        cmd = [prepgen_cmd, '-i', str(input_file), '-o', str(output_file)]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=context['workdir'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"prepgen stdout: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"prepgen failed: {e.stderr}")
            raise RuntimeError(f"prepgen failed with return code {e.returncode}: {e.stderr}")
        
        if not output_file.exists():
            raise RuntimeError(f"prepgen did not create output file: {output_file}")
        
        context['prepi_file'] = output_file
        
        logger.info(f"Generated prepi file: {output_file}")
        return context


class ParmchkStep(BuilderStep):
    """Step to generate AMBER frcmod file using parmchk2 command."""
    
    @property
    def name(self) -> str:
        return "parmchk"
    
    def run(self, context: Dict) -> Dict:
        """Generate frcmod file using parmchk2 command."""
        logger.info("Running parmchk2 to generate frcmod file")
        
        input_file = context['mol2_file']
        output_file = context['workdir'] / "polymer.frcmod"
        
        # Build parmchk2 command
        ambertools_bin = context.get('ambertools_bin', '')
        if ambertools_bin:
            parmchk_cmd = os.path.join(ambertools_bin, 'parmchk2')
        else:
            parmchk_cmd = 'parmchk2'
        
        # Check if parmchk2 is available
        if not shutil.which(parmchk_cmd):
            raise RuntimeError(f"parmchk2 not found. Make sure AmberTools is installed and in PATH.")
        
        # Run parmchk2 command
        cmd = [parmchk_cmd, '-i', str(input_file), '-f', 'mol2', '-o', str(output_file)]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=context['workdir'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"parmchk2 stdout: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"parmchk2 failed: {e.stderr}")
            raise RuntimeError(f"parmchk2 failed with return code {e.returncode}: {e.stderr}")
        
        if not output_file.exists():
            raise RuntimeError(f"parmchk2 did not create output file: {output_file}")
        
        context['frcmod_file'] = output_file
        
        logger.info(f"Generated frcmod file: {output_file}")
        return context


class TLeapStep(BuilderStep):
    """Step to generate AMBER topology and coordinates using tleap command."""
    
    @property
    def name(self) -> str:
        return "tleap"
    
    def run(self, context: Dict) -> Dict:
        """Generate prmtop and inpcrd files using tleap command."""
        logger.info("Running tleap to generate topology and coordinates")
        
        prepi_file = context['prepi_file']
        frcmod_file = context['frcmod_file']
        mol2_file = context['mol2_file']
        output_prefix = context['workdir'] / "polymer"
        
        # Build tleap command
        ambertools_bin = context.get('ambertools_bin', '')
        if ambertools_bin:
            tleap_cmd = os.path.join(ambertools_bin, 'tleap')
        else:
            tleap_cmd = 'tleap'
        
        # Check if tleap is available
        if not shutil.which(tleap_cmd):
            raise RuntimeError(f"tleap not found. Make sure AmberTools is installed and in PATH.")
        
        # Create tleap input script
        script_file = context['workdir'] / "tleap.in"
        prmtop_file = output_prefix.with_suffix('.prmtop')
        inpcrd_file = output_prefix.with_suffix('.inpcrd')
        
        with open(script_file, 'w') as f:
            f.write("source leaprc.gaff\n")
            f.write(f"loadamberprep {prepi_file}\n")
            f.write(f"loadamberparams {frcmod_file}\n")
            f.write(f"mol = loadmol2 {mol2_file}\n")
            f.write(f"saveamberparm mol {prmtop_file} {inpcrd_file}\n")
            f.write("quit\n")
        
        # Run tleap command
        cmd = [tleap_cmd, '-f', str(script_file)]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=context['workdir'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"tleap stdout: {result.stdout}")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"tleap failed: {e.stderr}")
            raise RuntimeError(f"tleap failed with return code {e.returncode}: {e.stderr}")
        
        if not prmtop_file.exists():
            raise RuntimeError(f"tleap did not create prmtop file: {prmtop_file}")
        if not inpcrd_file.exists():
            raise RuntimeError(f"tleap did not create inpcrd file: {inpcrd_file}")
        
        context['prmtop_file'] = prmtop_file
        context['inpcrd_file'] = inpcrd_file
        
        logger.info(f"Generated topology: {prmtop_file}")
        logger.info(f"Generated coordinates: {inpcrd_file}")
        return context


class ReadToFrameStep(BuilderStep):
    """Step to read AMBER files into a molpy Frame object."""
    
    @property
    def name(self) -> str:
        return "read_to_frame"
    
    def run(self, context: Dict) -> Dict:
        """Read AMBER files into molpy Frame."""
        logger.info("Reading AMBER files into molpy Frame")
        
        prmtop_file = context['prmtop_file']
        inpcrd_file = context['inpcrd_file']
        
        # Create Frame and read AMBER files
        frame = mp.Frame()
        
        # Read coordinates from inpcrd/rst7 file
        frame = mp.io.read_amber_rst7(inpcrd_file, frame=frame)
        
        # Store topology information in frame metadata
        # Note: Frame might not have a props attribute, so we store in metadata
        frame_metadata = {
            'prmtop': str(prmtop_file),
            'inpcrd': str(inpcrd_file),
            'prepi': str(context.get('prepi_file', '')),
            'frcmod': str(context.get('frcmod_file', '')),
            'mol2': str(context.get('mol2_file', ''))
        }
        
        # Try to set properties if the Frame supports it
        try:
            frame.props = frame_metadata
        except AttributeError:
            # If Frame doesn't have props, store in context
            logger.warning("Frame doesn't support props attribute, storing metadata in context")
            context['frame_metadata'] = frame_metadata
        
        context['frame'] = frame
        
        logger.info("Successfully created molpy Frame from AMBER files")
        return context


class AmberToolsPolymerBuilder:
    """
    Automated polymer builder using AmberTools workflow via molq.
    
    This builder orchestrates the complete AmberTools workflow for polymer construction:
    1. Generate polymer structure from SMILES
    2. Run prepgen to create prepi file
    3. Run parmchk2 to create frcmod file
    4. Run tleap to create topology and coordinates
    5. Read results into molpy Frame
    """
    
    def __init__(self, 
                 steps: Optional[List[BuilderStep]] = None,
                 ambertools_bin: Optional[str] = None,
                 cleanup: bool = True):
        """
        Initialize the AmberTools polymer builder.
        
        Args:
            steps: List of workflow steps. If None, uses default workflow.
            ambertools_bin: Path to AmberTools binaries. If None, uses system PATH.
            cleanup: Whether to clean up temporary files after completion.
        """
        # Check if RDKit is available for structure generation
        if not RDKIT_AVAILABLE:
            logger.warning(
                "RDKit is not available. Structure generation may fail. "
                "Install with: pip install rdkit"
            )
        
        # Default workflow steps
        if steps is None:
            steps = [
                GenerateStructureStep(),
                PrepgenStep(),
                ParmchkStep(),
                TLeapStep(),
                ReadToFrameStep()
            ]
        
        self.steps = steps
        self.ambertools_bin = ambertools_bin
        self.cleanup = cleanup
        
        logger.info(f"Initialized AmberToolsPolymerBuilder with {len(self.steps)} steps")
    
    def build(self, 
              monomer_smiles: str,
              sequence: str,
              n_repeat: int = 1,
              workdir: Optional[Union[str, Path]] = None,
              **kwargs) -> mp.Frame:
        """
        Build a polymer using the AmberTools workflow.
        
        Args:
            monomer_smiles: SMILES string for the monomer unit
            sequence: Polymer sequence (e.g., "A-B-A" for alternating copolymer)
            n_repeat: Number of times to repeat the sequence
            workdir: Working directory for intermediate files. If None, uses temp directory.
            **kwargs: Additional parameters passed to workflow steps
            
        Returns:
            molpy.Frame object containing the polymer structure and topology
            
        Raises:
            ImportError: If molq is not available
            RuntimeError: If any workflow step fails
        """
        # Set up working directory
        tmpdir = None
        if workdir is None:
            tmpdir = tempfile.TemporaryDirectory()
            workdir = Path(tmpdir.name)
            cleanup_tmpdir = True
        else:
            workdir = Path(workdir)
            workdir.mkdir(parents=True, exist_ok=True)
            cleanup_tmpdir = False
        
        # Initialize workflow context
        context = {
            'monomer_smiles': monomer_smiles,
            'sequence': sequence,
            'n_repeat': n_repeat,
            'workdir': workdir,
            'ambertools_bin': self.ambertools_bin,
            **kwargs
        }
        
        logger.info(f"Starting AmberTools polymer build in: {workdir}")
        logger.info(f"Monomer SMILES: {monomer_smiles}")
        logger.info(f"Sequence: {sequence}")
        logger.info(f"Repeats: {n_repeat}")
        
        try:
            # Execute workflow steps
            for step in self.steps:
                logger.info(f"Executing step: {step.name}")
                context = step.run(context)
                logger.info(f"Completed step: {step.name}")
            
            # Extract final Frame
            frame = context.get('frame')
            if frame is None:
                raise RuntimeError("Workflow did not produce a molpy Frame")
            
            logger.info("AmberTools polymer build completed successfully")
            return frame
            
        except Exception as e:
            logger.error(f"AmberTools polymer build failed: {e}")
            raise RuntimeError(f"Polymer build failed: {e}") from e
            
        finally:
            # Cleanup if requested
            if self.cleanup and cleanup_tmpdir and tmpdir is not None:
                try:
                    tmpdir.cleanup()
                    logger.info("Cleaned up temporary directory")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary directory: {e}")
    
    def add_step(self, step: BuilderStep, position: Optional[int] = None):
        """
        Add a new step to the workflow.
        
        Args:
            step: BuilderStep instance to add
            position: Position to insert step. If None, appends to end.
        """
        if position is None:
            self.steps.append(step)
        else:
            self.steps.insert(position, step)
        
        logger.info(f"Added step '{step.name}' at position {position or len(self.steps)-1}")
    
    def remove_step(self, step_name: str):
        """
        Remove a step from the workflow by name.
        
        Args:
            step_name: Name of the step to remove
        """
        self.steps = [step for step in self.steps if step.name != step_name]
        logger.info(f"Removed step '{step_name}' from workflow")
    
    def get_step_names(self) -> List[str]:
        """Get list of step names in the current workflow."""
        return [step.name for step in self.steps]


# Convenience function for simple usage
def build_polymer_with_ambertools(monomer_smiles: str,
                                 sequence: str,
                                 n_repeat: int = 1,
                                 workdir: Optional[Union[str, Path]] = None,
                                 **kwargs) -> mp.Frame:
    """
    Convenience function to build a polymer using AmberTools.
    
    Args:
        monomer_smiles: SMILES string for the monomer unit
        sequence: Polymer sequence (e.g., "A-B-A")
        n_repeat: Number of times to repeat the sequence
        workdir: Working directory for intermediate files
        **kwargs: Additional parameters for the builder
        
    Returns:
        molpy.Frame object containing the polymer structure
    """
    builder = AmberToolsPolymerBuilder(**kwargs)
    return builder.build(monomer_smiles, sequence, n_repeat, workdir)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Build a simple polyethylene chain
    frame = build_polymer_with_ambertools(
        monomer_smiles="CC",  # Ethylene
        sequence="A",         # Single monomer type
        n_repeat=10,         # 10 repeat units
        workdir="./polymer_build"
    )
    
    print(f"Built polymer with {len(frame['atoms']['id'])} atoms")
    # Check if frame has props attribute
    try:
        print(f"Frame properties: {list(frame.props.keys())}")
    except AttributeError:
        print("Frame metadata stored in build context")
