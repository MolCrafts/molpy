from molpy.builder.reacter_lammps import ReacterTemplateBuilder, ReactionTemplate, ReactionSite
from pathlib import Path
import molpy as mp
from molpy.io import read_pdb

"""
Example usage: Generate reaction templates for nylon-6,6 synthesis.

This example demonstrates the condensation reaction between
adipic acid and hexamethylenediamine to form nylon-6,6.

Based on LAMMPS examples: https://github.com/lammps/lammps/tree/develop/examples/PACKAGES/reaction/nylon%2C6-6_melt
"""
# Create adipic acid structure (HOOC-(CH2)4-COOH)
# For demonstration, assume this is loaded from PDB/MOL2 files
# adipic_acid_struct = mp.io.read_pdb("adipic_acid.pdb")
adipic_acid_struct = mp.AtomicStruct.from_frame(mp.io.read_pdb("adipicacid.pdb"), name="adipic_acid")

# Create hexamethylenediamine structure (H2N-(CH2)6-NH2)  
# hexamethylenediamine_struct = mp.io.read_pdb("hexamethylenediamine.pdb")
hexamethylenediamine_struct = mp.AtomicStruct.from_frame(mp.io.read_pdb("hexamethylenediamine.pdb"), name="hexamethylenediamine")

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

