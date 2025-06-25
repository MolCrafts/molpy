from pathlib import Path
import shutil
import molpy as mp
import xarray as xr
import molpy.pack as mpk
import molq
import tempfile
import numpy as np
from .base import Packer
from typing import Generator, Dict, Any


def map_region_to_packmol_definition(constraint):

    def box(region, not_flag):
        origin = region.origin
        upper = region.upper.tolist()
        flag = "outside" if not_flag else "inside"
        return f"{flag} box {origin[0]} {origin[1]} {origin[2]} {upper[0]} {upper[1]} {upper[2]}"
    
    def sphere(region, not_flag):
        origin = region.origin
        flag = "outside" if not_flag else "inside"
        return f"{flag} sphere {origin[0]} {origin[1]} {origin[2]} {region.radius}"

    if isinstance(constraint, mpk.AndConstraint):
        r1_cmd = map_region_to_packmol_definition(constraint.a)
        r2_cmd = map_region_to_packmol_definition(constraint.b)
        return f"  {r1_cmd} \n  {r2_cmd}"

    if isinstance(constraint, mpk.InsideBoxConstraint):
        return box(constraint.region, False)
    elif isinstance(constraint, mpk.InsideSphereConstraint):
        return sphere(constraint.region, False)
    elif isinstance(constraint, mpk.OutsideBoxConstraint):
        return box(constraint.region, True)
    elif isinstance(constraint, mpk.OutsideSphereConstraint):
        return sphere(constraint.region, True)
    elif isinstance(constraint, mpk.MinDistanceConstraint):
        pass
    else:
        raise NotImplementedError(
            f"Packmol does not support constraint type {type(constraint)}"
        )

class Packmol(Packer):

    def __init__(self, workdir: Path = Path.cwd(), executable: str = "packmol"):
        super().__init__()
        self.workdir = workdir
        self.cmd = executable
        self.optimizer = shutil.which(self.cmd)
        if self.optimizer is None:
            raise FileNotFoundError("Packmol not found in PATH")

    def generate_input(self, targets, max_steps, seed: int):

        self.intermediate_files = [
            ".optimized.pdb",
            ".packmol.inp",
            ".packmol.out",
        ]

        lines = []
        lines.append("tolerance 2.0")
        lines.append("filetype pdb")
        lines.append("output .optimized.pdb")
        lines.append(f"seed {seed}")
        for target in targets:
            frame = target.frame
            number = target.number
            constraint = target.constraint

            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")

            mp.io.write_pdb(Path(tmpfile.name), frame)

            lines.append(f"structure {tmpfile.name}")
            lines.append(f"  number {number}")
            lines.append(
                map_region_to_packmol_definition(constraint)
            )
            lines.append(f"end structure")

        with open(".packmol.inp", "w") as f:
            f.write("\n".join(lines))

    def remove_input(self):
        for file in self.intermediate_files:
            Path(file).unlink()

    @molq.local
    def pack(self, targets=None, max_steps: int = 1000, seed: int | None = None) -> Generator[Dict, Any, mp.Frame]:
        if targets is None:
            targets = self.targets
        if seed is None:
            seed = 4628
        self.generate_input(targets, max_steps, seed)
        yield {
            "cmd": "packmol < .packmol.inp > .packmol.out",
            "cwd": self.workdir,
            "cleanup_temp_files": True,
            "block": True,
        }
        optimized_frame = mp.io.read_pdb(Path(".optimized.pdb"))
        # self.remove_input()
        
        # Calculate total atoms for each target instance
        target_atoms_count = []
        for target in targets:
            for _ in range(target.number):
                # Get the actual number of atoms from the atoms dataset
                atoms_dataset = target.frame["atoms"]
                # Find the atoms dimension (usually 'atoms_id')
                atoms_dim = None
                for dim_name in atoms_dataset.dims:
                    if str(dim_name).endswith('_id') or 'atom' in str(dim_name).lower():
                        atoms_dim = dim_name
                        break
                if atoms_dim is None:
                    atoms_dim = list(atoms_dataset.dims.keys())[0]
                
                atoms_in_target = atoms_dataset.sizes[atoms_dim]
                target_atoms_count.append(atoms_in_target)
        
        # Calculate cumulative atom offsets
        atom_offsets = np.concatenate([[0], np.cumsum(target_atoms_count)])
        
        frames = []
        current_instance = 0
        bond_id_counter = 0
        angle_id_counter = 0
        dihedral_id_counter = 0
        
        for target in targets:
            for i in range(target.number):
                # Copy the entire frame for this instance
                instance_frame = mp.Frame()
                
                # Calculate current atom offset
                current_offset = atom_offsets[current_instance]
                
                # Process atoms
                atoms = target.frame["atoms"].copy()
                
                # Add molid if not present
                if "molid" not in atoms.data_vars:
                    main_dim = None
                    for dim_name in atoms.dims:
                        if str(dim_name).endswith('_id') or 'atom' in str(dim_name).lower():
                            main_dim = dim_name
                            break
                    if main_dim is None:
                        main_dim = list(atoms.dims.keys())[0] if atoms.dims else 'atoms_id'
                    atoms = atoms.assign(molid=(main_dim, np.full(atoms.sizes[main_dim], current_instance + 1)))
                
                # Update atom IDs with offset
                if "id" in atoms.data_vars:
                    # Always assign sequential IDs starting from current_offset + 1
                    # regardless of the original ID values
                    new_ids = np.arange(current_offset + 1, current_offset + len(atoms["id"]) + 1)
                    atoms = atoms.assign(id=(atoms["id"].dims, new_ids))
                if "molid" in atoms.data_vars:
                    atoms = atoms.assign(molid=(atoms["molid"].dims, np.full(atoms.sizes[atoms["molid"].dims[0]], current_instance + 1)))
                # Note: 'type' field contains string atom types (e.g., 'ca', 'c3') which should not be offset
                # Only type_id fields need offsetting, but we don't handle that here since it's handled 
                # in the bonds/angles/dihedrals sections below
                
                instance_frame["atoms"] = atoms
                
                # Process bonds if present
                if "bonds" in target.frame:
                    bonds = target.frame["bonds"].copy()
                    if "id" in bonds.data_vars:
                        new_bond_ids = np.arange(bond_id_counter + 1, bond_id_counter + len(bonds["id"]) + 1)
                        bonds = bonds.assign(id=(bonds["id"].dims, new_bond_ids))
                        bond_id_counter += len(bonds["id"])
                    if "i" in bonds.data_vars:
                        bonds = bonds.assign(i=(bonds["i"].dims, bonds["i"].values + current_offset))
                    if "j" in bonds.data_vars:
                        bonds = bonds.assign(j=(bonds["j"].dims, bonds["j"].values + current_offset))
                    instance_frame["bonds"] = bonds
                
                # Process angles if present
                if "angles" in target.frame:
                    angles = target.frame["angles"].copy()
                    if "id" in angles.data_vars:
                        new_angle_ids = np.arange(angle_id_counter + 1, angle_id_counter + len(angles["id"]) + 1)
                        angles = angles.assign(id=(angles["id"].dims, new_angle_ids))
                        angle_id_counter += len(angles["id"])
                    if "i" in angles.data_vars:
                        angles = angles.assign(i=(angles["i"].dims, angles["i"].values + current_offset))
                    if "j" in angles.data_vars:
                        angles = angles.assign(j=(angles["j"].dims, angles["j"].values + current_offset))
                    if "k" in angles.data_vars:
                        angles = angles.assign(k=(angles["k"].dims, angles["k"].values + current_offset))
                    instance_frame["angles"] = angles
                
                # Process dihedrals if present
                if "dihedrals" in target.frame:
                    dihedrals = target.frame["dihedrals"].copy()
                    if "id" in dihedrals.data_vars:
                        new_dihedral_ids = np.arange(dihedral_id_counter + 1, dihedral_id_counter + len(dihedrals["id"]) + 1)
                        dihedrals = dihedrals.assign(id=(dihedrals["id"].dims, new_dihedral_ids))
                        dihedral_id_counter += len(dihedrals["id"])
                    if "i" in dihedrals.data_vars:
                        dihedrals = dihedrals.assign(i=(dihedrals["i"].dims, dihedrals["i"].values + current_offset))
                    if "j" in dihedrals.data_vars:
                        dihedrals = dihedrals.assign(j=(dihedrals["j"].dims, dihedrals["j"].values + current_offset))
                    if "k" in dihedrals.data_vars:
                        dihedrals = dihedrals.assign(k=(dihedrals["k"].dims, dihedrals["k"].values + current_offset))
                    if "l" in dihedrals.data_vars:
                        dihedrals = dihedrals.assign(l=(dihedrals["l"].dims, dihedrals["l"].values + current_offset))
                    instance_frame["dihedrals"] = dihedrals
                
                frames.append(instance_frame)
                current_instance += 1
            
            # Note: We don't need to track n_atom_types anymore since atom 'type' fields 
            # are now strings (like 'ca', 'c3') rather than integer IDs that need offsetting
        
        # Concatenate all frames using direct xarray operations
        all_atoms = []
        all_bonds = []
        all_angles = []
        all_dihedrals = []
        
        for frame in frames:
            all_atoms.append(frame["atoms"])
            if "bonds" in frame:
                all_bonds.append(frame["bonds"])
            if "angles" in frame:
                all_angles.append(frame["angles"])
            if "dihedrals" in frame:
                all_dihedrals.append(frame["dihedrals"])
        
        # Create final frame
        final_frame = mp.Frame()
        
        # Concatenate atoms
        if all_atoms:
            combined_atoms = xr.concat(all_atoms, dim="atoms_id")
            # Update coordinates from optimized positions
            opt_coords = optimized_frame["atoms"]["xyz"]  # shape: (n_atoms, 3)
            combined_atoms["xyz"] = opt_coords
            final_frame["atoms"] = combined_atoms
        
        # Concatenate bonds
        if all_bonds:
            combined_bonds = xr.concat(all_bonds, dim="bonds_id")
            final_frame["bonds"] = combined_bonds
            
        # Concatenate angles
        if all_angles:
            combined_angles = xr.concat(all_angles, dim="angles_id")
            final_frame["angles"] = combined_angles
            
        # Concatenate dihedrals
        if all_dihedrals:
            combined_dihedrals = xr.concat(all_dihedrals, dim="dihedrals_id")
            final_frame["dihedrals"] = combined_dihedrals
        
        return final_frame