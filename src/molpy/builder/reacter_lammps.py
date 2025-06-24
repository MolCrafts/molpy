from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
import tempfile
from dataclasses import dataclass, field
import molq
import molpy as mp
from ..core.wrapper import Wrapper

logger = logging.getLogger(__name__)


@dataclass
class ReactionSite:
    """
    Defines a reaction site with anchor atoms and edge connections.
    """
    anchor_atom: int  # Main anchor atom index
    edge_atoms: List[int] = field(default_factory=list)  # Atoms that will be deleted/modified
    bond_target: Optional[int] = None  # Target atom for new bond formation


class ReactionTemplate(Wrapper):
    """
    Wrapper for AtomicStructure with reaction-specific functionality.
    Manages reaction sites, anchor points, and template extraction.
    """
    
    def __init__(
        self, 
        struct: mp.AtomicStructure, 
        reaction_sites: Optional[Dict[str, ReactionSite]] = None,
        **kwargs
    ):
        super().__init__(struct, **kwargs)
        self.reaction_sites = reaction_sites or {}
    
    def add_reaction_site(self, name: str, site: ReactionSite):
        """Add a reaction site to the template."""
        self.reaction_sites[name] = site
    
    def get_topology(self):
        """Get topology graph from the wrapped structure."""
        return self._wrapped.get_topology()
    
    def extract_fragment(self, start: int, end: int) -> 'ReactionTemplate':
        """
        Extract molecular fragment between two anchor points.
        
        Args:
            start: Starting atom index
            end: Ending atom index
            
        Returns:
            New ReactionTemplate with extracted fragment
        """
        topology = self.get_topology()
        main_chain, branches = get_main_chain_and_branches(topology, start, end)
        template_atoms = set(main_chain + branches)
        
        # Extract substructure
        extracted_struct = self._extract_substructure(template_atoms)
        
        # Create new template with adjusted reaction sites
        new_sites = {}
        for site_name, site in self.reaction_sites.items():
            if site.anchor_atom in template_atoms:
                # Map old indices to new indices
                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(template_atoms))}
                new_anchor = old_to_new.get(site.anchor_atom)
                new_edges = [old_to_new[edge] for edge in site.edge_atoms if edge in old_to_new]
                new_target = old_to_new.get(site.bond_target) if site.bond_target else None
                
                if new_anchor is not None:
                    new_sites[site_name] = ReactionSite(
                        anchor_atom=new_anchor,
                        edge_atoms=new_edges,
                        bond_target=new_target
                    )
        
        return ReactionTemplate(extracted_struct, new_sites)
    
    def _extract_substructure(self, atom_indices: set) -> mp.AtomicStructure:
        """Extract substructure containing only specified atoms."""
        new_struct = mp.AtomicStructure(name=f"{self._wrapped.get('name', 'molecule')}_template")
        
        # Map old indices to new indices
        index_map = {}
        sorted_indices = sorted(atom_indices)
        
        # Add atoms
        for new_idx, old_idx in enumerate(sorted_indices):
            atom = self._wrapped.atoms[old_idx]
            new_atom = atom.copy()
            new_atom["id"] = new_idx
            new_struct.atoms.add(new_atom)
            index_map[old_idx] = new_idx
        
        # Add bonds between extracted atoms
        for bond in self._wrapped.bonds:
            atom1_idx = bond.itom.get("id", bond.itom.get("index"))
            atom2_idx = bond.jtom.get("id", bond.jtom.get("index"))
            
            if atom1_idx in index_map and atom2_idx in index_map:
                new_bond = mp.Bond(
                    new_struct.atoms[index_map[atom1_idx]],
                    new_struct.atoms[index_map[atom2_idx]],
                    **{k: v for k, v in bond.items() if k not in ['itom', 'jtom']}
                )
                new_struct.bonds.add(new_bond)
        
        return new_struct


def get_main_chain_and_branches(g: mp.Topology, start: int, end: int):
    """
    Extract main chain and branches from topology graph.
    
    Args:
        g: Topology graph
        start: Starting atom index
        end: Ending atom index
        
    Returns:
        Tuple of (main_chain_atoms, branch_atoms)
    """
    # 1. Compute shortest path between start and end
    path = g.get_shortest_paths(start, to=end, output="vpath")[0]
    main_chain = set(path)

    # 2. Initialize visited with all main chain atoms
    visited = set(path)
    branch_atoms = set()

    # 3. Start from second atom in the path (exclude start atom's branches)
    for v in path[1:]:
        for n in g.neighbors(v):
            if n not in visited:
                # Collect entire connected component starting from n
                subtree = g.subcomponent(n, mode="ALL")
                branch_atoms.update([x for x in subtree if x not in main_chain])
                visited.update(subtree)

    return list(path), list(branch_atoms)


class ReacterTemplateBuilder:
    """
    LAMMPS-compatible reaction template builder for automated polymer synthesis.
    
    This class automates the generation of pre- and post-reaction molecular templates
    for use with LAMMPS fix bond/react functionality.
    """
    
    def __init__(
        self,
        template1: ReactionTemplate,
        template2: ReactionTemplate,
        bond_changes: List[Tuple[str, str, str, str]] = [],  # (site1, site2, atom1, atom2)
        atoms_to_delete: List[Tuple[str, str]] = [],  # (template_name, site_name)
        build_dir: Path = Path("reacter_build"),
        conda_env: str = "AmberTools25"
    ):
        """
        Initialize the reaction template builder.
        
        Args:
            template1: First reaction template
            template2: Second reaction template
            bond_changes: List of bond formation specifications
            atoms_to_delete: List of atoms to delete during reaction
            build_dir: Directory for output files
            conda_env: Conda environment for AmberTools
        """
        self.template1 = template1
        self.template2 = template2
        self.bond_changes = bond_changes
        self.atoms_to_delete = atoms_to_delete
        self.build_dir = Path(build_dir)
        self.conda_env = conda_env
        self.build_dir.mkdir(parents=True, exist_ok=True)
        
        # Store atom offset for merged structure
        self.atom_offset = 0

    def extract_templates(self) -> Tuple[ReactionTemplate, ReactionTemplate]:
        """
        Extract reaction templates from structures.
        
        Returns:
            Tuple of (template1, template2) as ReactionTemplate objects
        """
        logger.info("Extracting reaction templates")
        
        # Templates are already extracted, just return them
        logger.info(f"Template 1: {len(self.template1.atoms)} atoms")
        logger.info(f"Template 2: {len(self.template2.atoms)} atoms")
        
        return self.template1, self.template2

    def merge_structs(self) -> mp.AtomicStructure:
        """
        Merge template1 and template2 into a single structure.
        
        Offsets atom IDs from template2 to prevent collisions and adds
        metadata to track molecular origins.
        
        Returns:
            Merged AtomicStructure
        """
        logger.info("Merging templates")
        
        merged = mp.AtomicStructure(name="merged_pre_reaction")
        self.atom_offset = len(self.template1.atoms)
        
        # Add atoms from template1
        for atom in self.template1.atoms:
            new_atom = atom.copy()
            new_atom["molecule_origin"] = 1
            merged.atoms.add(new_atom)
        
        # Add atoms from template2 with offset
        for atom in self.template2.atoms:
            new_atom = atom.copy()
            new_atom["id"] = atom.get("id", 0) + self.atom_offset
            new_atom["molecule_origin"] = 2
            merged.atoms.add(new_atom)
        
        # Add bonds from template1
        for bond in self.template1.bonds:
            merged.bonds.add(bond.copy())
        
        # Add bonds from template2 with offset
        for bond in self.template2.bonds:
            atom1_id = bond.itom.get("id", 0) + self.atom_offset
            atom2_id = bond.jtom.get("id", 0) + self.atom_offset
            
            # Find atoms in merged structure
            atom1 = next(a for a in merged.atoms if a.get("id") == atom1_id)
            atom2 = next(a for a in merged.atoms if a.get("id") == atom2_id)
            
            new_bond = mp.Bond(atom1, atom2, **{k: v for k, v in bond.items() if k not in ['itom', 'jtom']})
            merged.bonds.add(new_bond)
        
        logger.info(f"Merged structure: {len(merged.atoms)} atoms, {len(merged.bonds)} bonds")
        return merged

    def apply_reaction(self, struct: mp.AtomicStructure) -> mp.AtomicStructure:
        """
        Apply bond creations and atom deletions to simulate post-reaction state.
        
        Args:
            struct: Pre-reaction structure
            
        Returns:
            Post-reaction AtomicStructure
        """
        logger.info("Applying reaction transformations")
        
        post_struct = struct.copy()
        post_struct["name"] = "post_reaction"
        
        # Create new bonds based on reaction site specifications
        for site1_name, site2_name, atom1_type, atom2_type in self.bond_changes:
            site1 = self.template1.reaction_sites.get(site1_name)
            site2 = self.template2.reaction_sites.get(site2_name)
            
            if site1 and site2:
                # Get actual atom IDs
                atom1_id = site1.anchor_atom if atom1_type == "anchor" else site1.bond_target
                if atom2_type == "anchor":
                    atom2_id = site2.anchor_atom + self.atom_offset
                else:
                    atom2_id = site2.bond_target + self.atom_offset if site2.bond_target is not None else None
                
                if atom1_id is not None and atom2_id is not None:
                    atom1 = next((a for a in post_struct.atoms if a.get("id") == atom1_id), None)
                    atom2 = next((a for a in post_struct.atoms if a.get("id") == atom2_id), None)
                    
                    if atom1 and atom2:
                        new_bond = mp.Bond(atom1, atom2, type="single")
                        post_struct.bonds.add(new_bond)
                        logger.info(f"Created bond between atoms {atom1_id} and {atom2_id}")
        
        # Delete specified atoms
        atoms_to_remove = []
        for template_name, site_name in self.atoms_to_delete:
            if template_name == "template1":
                site = self.template1.reaction_sites.get(site_name)
                if site:
                    for edge_atom in site.edge_atoms:
                        atom = next((a for a in post_struct.atoms if a.get("id") == edge_atom), None)
                        if atom:
                            atoms_to_remove.append(atom)
            elif template_name == "template2":
                site = self.template2.reaction_sites.get(site_name)
                if site:
                    for edge_atom in site.edge_atoms:
                        adjusted_id = edge_atom + self.atom_offset
                        atom = next((a for a in post_struct.atoms if a.get("id") == adjusted_id), None)
                        if atom:
                            atoms_to_remove.append(atom)
        
        # Remove bonds involving deleted atoms
        bonds_to_remove = []
        for bond in post_struct.bonds:
            if any(bond.itom.get("id") == atom.get("id") or 
                   bond.jtom.get("id") == atom.get("id") for atom in atoms_to_remove):
                bonds_to_remove.append(bond)
        
        for bond in bonds_to_remove:
            post_struct.bonds.remove(bond)
        
        for atom in atoms_to_remove:
            post_struct.atoms.remove(atom)
            logger.info(f"Deleted atom {atom.get('id')}")
        
        return post_struct

    def assign_forcefield(self, struct: mp.AtomicStructure, name: str):
        """
        Assign GAFF forcefield parameters using AmberTools builder.
        
        Args:
            struct: Structure to parameterize
            name: Base name for output files
        """
        logger.info(f"Assigning forcefield for {name}")
        
        from .ambertools import AntechamberStep, ParmchkStep
        
        workdir = self.build_dir / name
        workdir.mkdir(parents=True, exist_ok=True)
        
        # Write structure to MOL2 format first
        mol2_path = workdir / f"{name}.mol2"
        self._write_mol2(struct, mol2_path)
        
        # Use AmberTools steps
        antechamber_step = AntechamberStep(workdir, self.conda_env)
        parmchk_step = ParmchkStep(workdir, self.conda_env)
        
        # Run antechamber
        antechamber_step.run(
            name=name,
            monomer=struct,  # Adapt as needed
            net_charge=0.0,
            forcefield="gaff",
            charge_type="bcc"
        )
        
        # Run parmchk2
        parmchk_step.run(name)
        
        logger.info(f"Generated forcefield files for {name}")

    def _write_mol2(self, struct: mp.AtomicStructure, filepath: Path):
        """Write structure in MOL2 format."""
        with open(filepath, 'w') as f:
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{struct.get('name', 'molecule')}\n")
            f.write(f"{len(struct.atoms)} {len(struct.bonds)} 0 0 0\n")
            f.write("SMALL\n")
            f.write("NO_CHARGES\n\n")
            
            f.write("@<TRIPOS>ATOM\n")
            for i, atom in enumerate(struct.atoms):
                atom_id = atom.get("id", i + 1)
                name = atom.get("name", f"A{atom_id}")
                element = atom.get("element", "C")
                x, y, z = atom.get("xyz", [0.0, 0.0, 0.0])
                f.write(f"{atom_id:7d} {name:8s} {x:10.4f} {y:10.4f} {z:10.4f} {element:4s} 1    MOL 0.0000\n")
            
            f.write("\n@<TRIPOS>BOND\n")
            for i, bond in enumerate(struct.bonds):
                bond_id = i + 1
                atom1_id = bond.itom.get("id", 1)
                atom2_id = bond.jtom.get("id", 2)
                bond_type = bond.get("type", "1")
                f.write(f"{bond_id:6d} {atom1_id:6d} {atom2_id:6d} {bond_type:2s}\n")

    def export_lammps_template(self, struct: mp.AtomicStructure, name: str) -> Path:
        """
        Export structure as LAMMPS molecule template using built-in writer.
        
        Args:
            struct: Structure to export
            name: Template name
            
        Returns:
            Path to generated template file
        """
        logger.info(f"Exporting LAMMPS template for {name}")
        
        template_path = self.build_dir / f"{name}.template"
        
        # Convert to frame format
        frame = struct.to_frame()
        
        # Use molpy's built-in LAMMPS molecule writer
        mp.io.write_lammps_molecule(template_path, frame)
        
        logger.info(f"LAMMPS template written to {template_path}")
        return template_path

    def generate_mapping(self, struct_pre: mp.AtomicStructure, struct_post: mp.AtomicStructure) -> Dict[str, str]:
        """
        Generate atom mapping file for LAMMPS fix bond/react.
        
        According to LAMMPS documentation, the mapping file format is:
        N
        template-ID1 pre-atom-ID post-atom-ID
        template-ID2 pre-atom-ID post-atom-ID
        ...
        
        Args:
            struct_pre: Pre-reaction structure
            struct_post: Post-reaction structure
            
        Returns:
            Dictionary with mapping information
        """
        logger.info("Generating LAMMPS-compatible atom mapping")
        
        mapping = {}
        
        # Create mapping based on atom IDs that weren't deleted
        deleted_ids = set()
        for template_name, site_name in self.atoms_to_delete:
            if template_name == "template1":
                site = self.template1.reaction_sites.get(site_name)
                if site:
                    deleted_ids.update(site.edge_atoms)
            elif template_name == "template2":
                site = self.template2.reaction_sites.get(site_name)
                if site:
                    deleted_ids.update([edge + self.atom_offset for edge in site.edge_atoms])
        
        mapping_entries = []
        for atom_pre in struct_pre.atoms:
            atom_id = atom_pre.get("id")
            if atom_id not in deleted_ids:
                # Find corresponding atom in post structure
                atom_post = next((a for a in struct_post.atoms if a.get("id") == atom_id), None)
                if atom_post:
                    mapping_entries.append((atom_id, atom_post.get("id")))
        
        # Write mapping file in LAMMPS format
        mapping_path = self.build_dir / "mapping.txt"
        with open(mapping_path, 'w') as f:
            f.write(f"{len(mapping_entries)}\n")
            for pre_id, post_id in mapping_entries:
                # LAMMPS format: template-ID pre-atom-ID post-atom-ID
                f.write(f"reaction_template {pre_id} {post_id}\n")
        
        logger.info(f"Generated LAMMPS mapping for {len(mapping_entries)} atoms")
        logger.info(f"Mapping saved to {mapping_path}")
        
        return {"mapping_file": str(mapping_path), "entries": str(len(mapping_entries))}

    def build_all(self):
        """
        Execute the complete workflow for reaction template generation.
        
        Performs the following steps:
        1. Extract molecular templates
        2. Merge structures  
        3. Generate force fields for pre- and post-reaction states
        4. Export LAMMPS templates
        5. Generate atom mapping
        6. Write all outputs to build directory
        """
        logger.info("Starting complete reaction template build workflow")
        
        try:
            # Extract templates (already done in constructor)
            logger.info("Step 1/7: Templates ready...")
            template1, template2 = self.extract_templates()
            
            # Merge structures
            logger.info("Step 2/7: Merging structures...")
            pre_struct = self.merge_structs()
            
            # Apply reaction to get post-reaction structure
            logger.info("Step 3/7: Applying reaction...")
            post_struct = self.apply_reaction(pre_struct)
            
            # Assign force fields
            logger.info("Step 4/7: Assigning force field (pre-reaction)...")
            self.assign_forcefield(pre_struct, name="pre")
            
            logger.info("Step 5/7: Assigning force field (post-reaction)...")
            self.assign_forcefield(post_struct, name="post")
            
            # Export LAMMPS templates
            logger.info("Step 6/7: Exporting LAMMPS templates...")
            pre_template_path = self.export_lammps_template(pre_struct, name="pre")
            post_template_path = self.export_lammps_template(post_struct, name="post")
            
            # Generate mapping
            logger.info("Step 7/7: Generating atom mapping...")
            mapping = self.generate_mapping(pre_struct, post_struct)
            
            # Write summary
            summary_path = self.build_dir / "build_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("LAMMPS Reaction Template Build Summary\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Pre-reaction atoms: {len(pre_struct.atoms)}\n")
                f.write(f"Post-reaction atoms: {len(post_struct.atoms)}\n")
                f.write(f"Bonds created: {len(self.bond_changes)}\n")
                f.write(f"Atoms deleted: {len(self.atoms_to_delete)}\n")
                f.write(f"Mapping entries: {mapping['entries']}\n\n")
                f.write("Generated files:\n")
                f.write(f"- Pre-reaction template: {pre_template_path}\n")
                f.write(f"- Post-reaction template: {post_template_path}\n")
                f.write(f"- LAMMPS mapping file: {mapping['mapping_file']}\n")
                f.write(f"- Force field files: pre/ and post/ directories\n\n")
                f.write("LAMMPS fix bond/react usage:\n")
                f.write("fix ID group-ID bond/react stabilization yes nvt_grp 0.1 react_one all &\n")
                f.write("    initiate 1 0 5.0 template1 pre.template template2 post.template mapping.txt\n")
            
            logger.info("Workflow completed successfully!")
            logger.info(f"All files written to: {self.build_dir}")
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            raise

def create_nylon66_example():
    """
    Example usage: Generate reaction templates for nylon-6,6 synthesis.
    
    This example demonstrates the condensation reaction between
    adipic acid and hexamethylenediamine to form nylon-6,6.
    
    Based on LAMMPS examples: https://github.com/lammps/lammps/tree/develop/examples/PACKAGES/reaction/nylon%2C6-6_melt
    """
    import molpy as mp
    
    # Create adipic acid structure (HOOC-(CH2)4-COOH)
    # For demonstration, assume this is loaded from PDB/MOL2 files
    # adipic_acid_struct = mp.io.read_pdb("adipic_acid.pdb")
    adipic_acid_struct = mp.AtomicStructure(name="adipic_acid")
    
    # Create hexamethylenediamine structure (H2N-(CH2)6-NH2)  
    # hexamethylenediamine_struct = mp.io.read_pdb("hexamethylenediamine.pdb")
    hexamethylenediamine_struct = mp.AtomicStructure(name="hexamethylenediamine")
    
    # Define reaction sites for adipic acid
    adipic_acid_sites = {
        "carboxyl_1": ReactionSite(
            anchor_atom=1,     # Carboxyl carbon
            edge_atoms=[2, 3], # OH atoms to be deleted
            bond_target=1      # Carbon that will form new bond
        ),
        "carboxyl_2": ReactionSite(
            anchor_atom=6,     # Other carboxyl carbon
            edge_atoms=[7, 8], # OH atoms to be deleted  
            bond_target=6      # Carbon that will form new bond
        )
    }
    
    # Define reaction sites for hexamethylenediamine
    diamine_sites = {
        "amino_1": ReactionSite(
            anchor_atom=0,     # Amino nitrogen
            edge_atoms=[14],   # H atom to be deleted
            bond_target=0      # Nitrogen that will form new bond
        ),
        "amino_2": ReactionSite(
            anchor_atom=7,     # Other amino nitrogen
            edge_atoms=[15],   # H atom to be deleted
            bond_target=7      # Nitrogen that will form new bond
        )
    }
    
    # Create reaction templates with site definitions
    adipic_template = ReactionTemplate(adipic_acid_struct, adipic_acid_sites)
    diamine_template = ReactionTemplate(hexamethylenediamine_struct, diamine_sites)
    
    # Extract fragments if needed (for full molecules, this step might be skipped)
    # adipic_fragment = adipic_template.extract_fragment(start=0, end=6)
    # diamine_fragment = diamine_template.extract_fragment(start=0, end=7)
    
    # Define reaction parameters for amide bond formation
    # In nylon-6,6 synthesis: R-COOH + H2N-R' -> R-CO-NH-R' + H2O
    
    builder = ReacterTemplateBuilder(
        template1=adipic_template,
        template2=diamine_template,
        bond_changes=[
            # (site1_name, site2_name, atom1_type, atom2_type)
            ("carboxyl_1", "amino_1", "anchor", "anchor"),  # Form C-N amide bond
        ],
        atoms_to_delete=[
            # (template_name, site_name) - deletes edge_atoms from the site
            ("template1", "carboxyl_1"),  # Delete OH from COOH  
            ("template2", "amino_1"),     # Delete H from NH2 (forms H2O)
        ],
        build_dir=Path("nylon66_templates"),
        conda_env="AmberTools25"
    )
    
    # Generate all reaction template files
    builder.build_all()
    
    print("Nylon-6,6 reaction templates generated!")
    print("Files created in nylon66_templates/:")
    print("- pre.template (pre-reaction LAMMPS molecule template)")
    print("- post.template (post-reaction LAMMPS molecule template)")
    print("- mapping.txt (LAMMPS-compatible atom mapping)")
    print("- pre/ (GAFF forcefield files for reactants)")
    print("- post/ (GAFF forcefield files for products)")
    print("- build_summary.txt (build summary)")
    print("\nTo use in LAMMPS:")
    print("fix react all bond/react stabilization yes nvt_grp 0.1 \\")
    print("    react_one all initiate 1 0 5.0 template1 pre.template \\")
    print("    template2 post.template mapping.txt")


def create_simple_reaction_example():
    """
    Simple example: A + B -> A-B reaction
    """
    import molpy as mp
    
    # Create simple molecules (this would normally be loaded from files)
    molA = mp.AtomicStructure(name="molecule_A")
    molB = mp.AtomicStructure(name="molecule_B")
    
    # Define simple reaction sites
    siteA = {"reactive": ReactionSite(anchor_atom=0, edge_atoms=[], bond_target=0)}
    siteB = {"reactive": ReactionSite(anchor_atom=0, edge_atoms=[], bond_target=0)}
    
    templateA = ReactionTemplate(molA, siteA)
    templateB = ReactionTemplate(molB, siteB)
    
    builder = ReacterTemplateBuilder(
        template1=templateA,
        template2=templateB,
        bond_changes=[("reactive", "reactive", "anchor", "anchor")],
        atoms_to_delete=[],  # No atoms deleted in this simple case
        build_dir=Path("simple_reaction_templates")
    )
    
    builder.build_all()
    print("Simple A + B -> A-B reaction templates generated!")


if __name__ == "__main__":
    # Run the nylon-6,6 example
    create_nylon66_example()
    
    # Uncomment to run simple example
    # create_simple_reaction_example()
