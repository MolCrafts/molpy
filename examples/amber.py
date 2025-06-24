# %%
import numpy as np
import molpy as mp
from pathlib import Path

data_path = Path("data/case1")

# %%
struct_h = mp.AtomicStructure.from_frame(mp.io.read_pdb(data_path / "H.pdb"), "H")
struct_n = mp.AtomicStructure.from_frame(mp.io.read_pdb(data_path / "N.pdb"), "N")
struct_m = mp.AtomicStructure.from_frame(mp.io.read_pdb(data_path / "M.pdb"), "M")
struct_p = mp.AtomicStructure.from_frame(mp.io.read_pdb(data_path / "P.pdb"), "P")
struct_t = mp.AtomicStructure.from_frame(mp.io.read_pdb(data_path / "T.pdb"), "T")
struct_tfsi = mp.AtomicStructure.from_frame(
    mp.io.read_pdb(data_path / "TFSI.pdb"), "TFSI"
)
struct_n["net_charge"] = 1
struct_m["net_charge"] = -1
struct_tfsi["net_charge"] = -1

monomer_h = mp.Monomer(
    struct_h, [mp.AnchorRule(name="tail", anchor=14, deletes=[15])]
)
monomer_n = mp.Monomer(
    struct_n,
    [
        mp.AnchorRule(name="head", anchor=3, deletes=[12]),
        mp.AnchorRule(name="tail", anchor=5, deletes=[11]),
    ],
)
monomer_m = mp.Monomer(
    struct_m,
    [
        mp.AnchorRule(name="head", anchor=20, deletes=[16]),
        mp.AnchorRule(name="tail", anchor=14, deletes=[9]),
    ],
)
monomer_p = mp.Monomer(
    struct_p,
    [
        mp.AnchorRule(name="head", anchor=0, deletes=[4]),
        mp.AnchorRule(name="tail", anchor=5, deletes=[7]),
    ],
)
monomer_t = mp.Monomer(
    struct_t, [mp.AnchorRule(name="head", anchor=1, deletes=[2])]
)

# %%
poly_builer = mp.builder.AmberToolsPolymerBuilder(workdir=data_path)
struct_poly = poly_builer.build(
    "polymer",
    [monomer_h, monomer_n, monomer_m, monomer_p, monomer_t],
    ["H", "N", "M", "P", "T"],
)
print(struct_poly.to_frame()["atoms"])

# %%
salt_builder = mp.builder.AmberToolsSaltBuilder(workdir=data_path)
struct_litfsi = salt_builder.build(
    "LiTFSI",
    mp.Monomer(struct_tfsi),
    "LI",
)

# %%
# === Load force field parameters ===
print("Loading AMBER force field parameters...")

# Read polymer force field
polymer_prmtop_path = data_path / "polymer" / "polymer.prmtop"
polymer_inpcrd_path = data_path / "polymer" / "polymer.inpcrd"
polymer_frame_with_ff = mp.io.read_amber(polymer_prmtop_path, polymer_inpcrd_path)
print(f"Loaded polymer force field from: {polymer_prmtop_path}")

# Debug: Check the original polymer frame from AMBER
print(f"\nDirect AMBER polymer frame atoms structure:")
print(polymer_frame_with_ff["atoms"])

# Read LiTFSI force field  
litfsi_prmtop_path = data_path / "LiTFSI" / "LiTFSI.prmtop"
litfsi_inpcrd_path = data_path / "LiTFSI" / "LiTFSI.inpcrd"
litfsi_frame_with_ff = mp.io.read_amber(litfsi_prmtop_path, litfsi_inpcrd_path)
print(f"Loaded LiTFSI force field from: {litfsi_prmtop_path}")

# Debug: Check the original LiTFSI frame from AMBER
print(f"\nDirect AMBER LiTFSI frame atoms structure:")
print(litfsi_frame_with_ff["atoms"])

# %%
# === Packing Section using molpy.pack ===
import molpy.pack as mpk

# Define simulation box parameters
box_size = np.array([40.0, 40.0, 40.0])  # 40 x 40 x 40 Angstrom box
box_origin = np.array([0.0, 0.0, 0.0])

# Create packing session
packing_session = mpk.Session(packer="packmol")

# Add polymer molecules to the system
n_polymer_chains = 5  # Number of polymer chains
polymer_constraint = mpk.InsideBoxConstraint(box_size, box_origin)
packing_session.add_target(
    frame=struct_poly.to_frame(),
    number=n_polymer_chains,
    constraint=polymer_constraint
)

# Add LiTFSI salt molecules
n_litfsi = 10  # Number of LiTFSI molecules
# Create constraint that keeps salt molecules inside box but with minimum distance from polymers
salt_constraint = (
    mpk.InsideBoxConstraint(box_size, box_origin)
)
packing_session.add_target(
    frame=struct_litfsi.to_frame(),
    number=n_litfsi,
    constraint=salt_constraint
)

# Perform packing optimization
print("Starting molecular packing optimization...")
packed_system = packing_session.optimize(max_steps=2000, seed=42)

print(f"Packing completed!")
print(f"Final system contains:")
print(f"- {n_polymer_chains} polymer chains")
print(f"- {n_litfsi} LiTFSI molecules")
print(f"- Total atoms: {len(packed_system['atoms']['id']) if 'id' in packed_system['atoms'] else 'unknown'}")

# Debug: Check packed system types before writing
print(f"\nPacked system atom types after packing:")
if "atoms" in packed_system and "type" in packed_system["atoms"]:
    print(f"Sample types: {packed_system['atoms']['type'].values[:10]}")
    print(f"Unique types: {np.unique(packed_system['atoms']['type'].values)}")
    print(f"Type of type values: {type(packed_system['atoms']['type'].values[0])}")
else:
    print("No type field found in packed system!")

# %%
# === Merge force field parameters ===
print("Merging force field parameters...")

# The force field information is already embedded in the bonds, angles, dihedrals datasets
# from the AMBER reader. We need to make sure the packed system uses the type_id fields
# for LAMMPS output rather than the string type names.

# Check what force field data we have from the AMBER files
print("Polymer frame structure:")
for key in polymer_frame_with_ff:
    if key in ['bonds', 'angles', 'dihedrals']:
        dataset = polymer_frame_with_ff[key]
        print(f"  {key}: {list(dataset.data_vars.keys())}")
        if 'type_id' in dataset.data_vars:
            print(f"    {key} type_id range: {dataset['type_id'].min().item()}-{dataset['type_id'].max().item()}")

print("\nLiTFSI frame structure:")
for key in litfsi_frame_with_ff:
    if key in ['bonds', 'angles', 'dihedrals']:
        dataset = litfsi_frame_with_ff[key]
        print(f"  {key}: {list(dataset.data_vars.keys())}")
        if 'type_id' in dataset.data_vars:
            print(f"    {key} type_id range: {dataset['type_id'].min().item()}-{dataset['type_id'].max().item()}")

# For now, we need to ensure that the packed system has proper type_id fields
# The packed system should already have the right structure from the packing process
print(f"\nPacked system structure:")
for key in packed_system:
    if key in ['bonds', 'angles', 'dihedrals']:
        dataset = packed_system[key]
        print(f"  {key}: {list(dataset.data_vars.keys())}")

# %%
# Save the packed system 
output_base = data_path / "packed_system"
lammps_data_path = Path(str(output_base) + ".data")  # Ensure it's a Path with .data extension

# Write only the LAMMPS data file (not the forcefield file)
# The force field parameters are embedded in the type_id fields of bonds/angles/dihedrals
mp.io.write_lammps_data(lammps_data_path, packed_system)
print(f"Packed system LAMMPS data file saved to: {lammps_data_path}")

# Also save as PDB for visualization
pdb_output_path = Path(str(output_base) + ".pdb")
mp.io.write_pdb(pdb_output_path, packed_system)
print(f"PDB file saved to: {pdb_output_path}")
