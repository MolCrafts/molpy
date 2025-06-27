from pathlib import Path
import logging
from dataclasses import dataclass, field
import molpy as mp
from ..core.wrapper import Wrapper

logger = logging.getLogger(__name__)


@dataclass
class ReactionTemplate:
    """
    Defines a reaction site with anchor atoms and edge connections.
    """
    init_atom: int  # Main anchor atom index
    edge_atom: int  # Atoms that will be deleted/modified

class ReactantWrapper(Wrapper):
    """
    Wrapper for AtomicStructure with reaction-specific functionality.
    Manages reaction sites, anchor points, and template extraction.
    """
    
    def __init__(
        self, 
        struct: mp.AtomicStructure, 
        **kwargs
    ):
        super().__init__(struct, **kwargs)
    
    def get_topology(self):
        """Get topology graph from the wrapped structure."""
        return self._wrapped.get_topology()
    
    def __call__(self, **kwargs):
        return self._wrapped(**kwargs)
    
    def extract_fragment(self, template: ReactionTemplate) -> 'ReactantWrapper':
        """
        Extract molecular fragment between two anchor points.
        
        Args:
            start: Starting atom index
            end: Ending atom index
            
        Returns:
            New ReactantWrapper with extracted fragment
        """
        topology = self.get_topology()
        start = template.edge_atom
        end = template.init_atom
        main_chain, branches = get_main_chain_and_branches(topology, start, end)
        template_atoms = set(main_chain + branches)
        
        # Extract substructure
        extracted_struct = self._extract_substructure(template_atoms)
        
        # Create new template with adjusted reaction sites
        # old_to_new_mapping = {j: i for i, j in enumerate(sorted(template_atoms))}
        # new_template = ReactionTemplate(
        #     init_atom=old_to_new_mapping[template.init_atom],
        #     edge_atom=old_to_new_mapping[template.edge_atom]
        # )
        
        return ReactantWrapper(extracted_struct)
    
    def _extract_substructure(self, atom_indices: set) -> mp.AtomicStructure:
        """Extract substructure containing only specified atoms."""
        new_struct = mp.AtomicStructure(name=f"{self._wrapped.get('name', 'molecule')}_template")
        
        # Map old atom to new atom
        atom_map = {}
        
        # Add atoms
        for old_idx in atom_indices:
            atom = self._wrapped.atoms[old_idx]
            new_atom = atom.copy()
            new_struct.atoms.add(new_atom)
            atom_map[atom] = new_atom
        
        # Add bonds between extracted atoms
        for bond in self._wrapped.bonds:
            atom1 = bond.itom
            atom2 = bond.jtom
            if atom1 in atom_map and atom2 in atom_map:
                new_bond = mp.Bond(
                    atom_map[atom1],
                    atom_map[atom2],
                    **{k: v for k, v in bond.items()}
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
    from collections import deque
    def get_branch_subtree(g: mp.Topology, start: int, excluded: set):
        visited = set()
        queue = deque([start])
        while queue:
            v = queue.popleft()
            if v in visited or v in excluded:
                continue
            visited.add(v)
            for n in g.neighbors(v):
                if n not in visited and n not in excluded:
                    queue.append(n)
        return visited

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
                subtree = get_branch_subtree(g, n, main_chain)
                branch_atoms.update([x for x in subtree if x not in main_chain])
                visited.update(subtree)

    return list(path), list(branch_atoms)


class ReacterBuilder:
    """
    LAMMPS-compatible reaction template builder for automated polymer synthesis.
    
    This class automates the generation of pre- and post-reaction molecular templates
    for use with LAMMPS fix bond/react functionality.
    """
    
    def __init__(
        self,
        workdir: Path,
        typifier,
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
        self.workdir = workdir
        self.typifier = typifier
        self.workdir.mkdir(parents=True, exist_ok=True)


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
        
        template_path = self.workdir / f"{name}.template"
        
        # Convert to frame format
        frame = struct.to_frame()
        # Use molpy's built-in LAMMPS molecule writer
        mp.io.write_lammps_molecule(template_path, frame)
        
        logger.info(f"LAMMPS template written to {template_path}")
        return template_path

    def generate_mapping(self, name, pre_struct: ReactantWrapper, post_struct: ReactantWrapper, inits: list[int], edges: list[int]) -> dict:
        """
        Generate LAMMPS mapping file for bond/react.
        
        Args:
            pre_struct: Pre-reaction structure
            post_struct: Post-reaction structure
            inits: List of initiator atom IDs
            edges: List of edge atom IDs
            
        Returns:
            Dictionary containing mapping information and file path
        """
        logger.info("Generating LAMMPS mapping file")
        
        # Get atoms from both structures
        pre_atoms = list(pre_struct.atoms)
        post_atoms = list(post_struct.atoms)
        
        # Generate 1:1 atom correspondence
        equivalences = []
        min_atoms = min(len(pre_atoms), len(post_atoms))
        
        for i in range(min_atoms):
            pre_id = pre_atoms[i].get("id", i + 1)
            post_id = post_atoms[i].get("id", i + 1)
            equivalences.append((pre_id, post_id))
        
        # Use provided initiator and edge atoms
        initiator_ids = inits if inits else [1, 2]  # Default to first two atoms
        edge_ids = edges if edges else []
        
        # Generate mapping file content
        mapping_content = self._generate_mapping_content(
            equivalences, initiator_ids, edge_ids
        )
        
        # Write mapping file
        mapping_file = self.workdir / f"{name}.map"
        with open(mapping_file, 'w') as f:
            f.write(mapping_content)
        
        logger.info(f"Generated mapping file: {mapping_file}")
        logger.info(f"Equivalences: {len(equivalences)}")
        logger.info(f"Initiator IDs: {initiator_ids}")
        logger.info(f"Edge IDs: {edge_ids}")
        
        return {
            'mapping_file': mapping_file,
            'entries': len(equivalences),
            'initiator_ids': initiator_ids,
            'edge_ids': edge_ids,
            'equivalences': equivalences
        }
    
    def _generate_mapping_content(self, equivalences, initiator_ids, edge_ids) -> str:
        """Generate the content of the mapping file."""
        lines = []
        
        # Header comment
        lines.append("# this is a map file")
        lines.append("")
        
        # Equivalences count
        lines.append(f"{len(equivalences)} equivalences")
        
        # Edge IDs count (optional)
        if edge_ids:
            lines.append(f"{len(edge_ids)} edgeIDs")
        
        lines.append("")
        
        # InitiatorIDs section (mandatory)
        lines.append("InitiatorIDs")
        lines.append("")
        for init_id in initiator_ids:
            lines.append(str(init_id))
        lines.append("")
        
        # EdgeIDs section (optional)
        if edge_ids:
            lines.append("EdgeIDs")
            lines.append("")
            for edge_id in edge_ids:
                lines.append(str(edge_id))
            lines.append("")
        
        # Equivalences section (mandatory)
        lines.append("Equivalences")
        lines.append("")
        for pre_id, post_id in equivalences:
            lines.append(f"{pre_id}   {post_id}")
        
        return "\n".join(lines)


    def build(self, name, pre_struct, post_struct, inits=[], edges=[]):
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
        self.typifier.typify(pre_struct)
        pre_template_path = self.export_lammps_template(pre_struct, name=pre_struct.get("name"))
        self.typifier.typify(post_struct)
        post_template_path = self.export_lammps_template(post_struct, name=post_struct.get("name"))
        mapping = self.generate_mapping(name, pre_struct, post_struct, inits, edges)
        print(f"- Pre-reaction template: {pre_template_path}\n")
        print(f"- Post-reaction template: {post_template_path}\n")
        return post_struct