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

        with open(self.workdir/".packmol.inp", "w") as f:
            f.write("\n".join(lines))

    def remove_input(self):
        for file in self.intermediate_files:
            (self.workdir/file).unlink()

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
        optimized_frame = mp.io.read_pdb(self.workdir/".optimized.pdb")
        self.remove_input()
        
        # Count atoms per instance
        target_atoms_count = []
        for target in targets:
            atoms_block = target.frame['atoms']
            n_atoms = len(atoms_block.get('id', atoms_block.get('xyz', [])))
            for _ in range(target.number):
                target_atoms_count.append(n_atoms)

        # Compute cumulative offsets
        atom_offsets = np.concatenate([[0], np.cumsum(target_atoms_count)])

        # Prepare lists for each block
        all_atoms = []
        all_bonds = []
        all_angles = []
        all_dihedrals = []

        current_instance = 0
        bond_id_counter = 0
        angle_id_counter = 0
        dihedral_id_counter = 0

        for target in targets:
            for inst in range(target.number):
                offset = atom_offsets[current_instance]
                frame_dict = {}

                # Process atoms block
                atoms = target.frame['atoms'].copy()
                n = len(atoms.get('id', atoms.get('xyz', [])))

                atoms['mol'] = np.full(n, current_instance + 1, dtype=int)

                # Reassign atom IDs sequentially
                if 'id' in atoms:
                    atoms['id'] = np.arange(offset + 1, offset + n + 1, dtype=int)

                frame_dict['atoms'] = atoms

                # Process bonds
                if 'bonds' in target.frame:
                    bonds = target.frame['bonds'].copy()
                    m = len(bonds.get('id', bonds.get('i', [])))
                    # IDs
                    if 'id' in bonds:
                        bonds['id'] = np.arange(bond_id_counter + 1, bond_id_counter + m + 1, dtype=int)
                        bond_id_counter += m
                    # Endpoints
                    for end in ('i', 'j'):
                        if end in bonds:
                            bonds[end] = bonds[end] + offset
                    frame_dict['bonds'] = bonds

                # Process angles
                if 'angles' in target.frame:
                    angles = target.frame['angles'].copy()
                    p = len(angles.get('id', angles.get('i', [])))
                    if 'id' in angles:
                        angles['id'] = np.arange(angle_id_counter + 1, angle_id_counter + p + 1, dtype=int)
                        angle_id_counter += p
                    for end in ('i', 'j', 'k'):
                        if end in angles:
                            angles[end] = angles[end] + offset
                    frame_dict['angles'] = angles

                # Process dihedrals
                if 'dihedrals' in target.frame:
                    dihedrals = target.frame['dihedrals'].copy()
                    q = len(dihedrals.get('id', dihedrals.get('i', [])))
                    if 'id' in dihedrals:
                        dihedrals['id'] = np.arange(dihedral_id_counter + 1, dihedral_id_counter + q + 1, dtype=int)
                        dihedral_id_counter += q
                    for end in ('i', 'j', 'k', 'l'):
                        if end in dihedrals:
                            dihedrals[end] = dihedrals[end] + offset
                    frame_dict['dihedrals'] = dihedrals

                # Collect instance frame
                all_atoms.append(frame_dict['atoms'])
                if 'bonds' in frame_dict:
                    all_bonds.append(frame_dict['bonds'])
                if 'angles' in frame_dict:
                    all_angles.append(frame_dict['angles'])
                if 'dihedrals' in frame_dict:
                    all_dihedrals.append(frame_dict['dihedrals'])

                current_instance += 1

        # Helper to concatenate dicts of arrays
        def concat_blocks(blocks):
            if not blocks:
                return {}
            keys = blocks[0].keys()
            combined = {}
            for key in keys:
                combined[key] = np.concatenate([blk[key] for blk in blocks], axis=0)
            return combined

        # Build final frame
        final_frame = mp.Frame()
        final_frame['atoms'] = concat_blocks(all_atoms)
        # Overwrite coordinates with optimized positions
        if 'xyz' in optimized_frame['atoms']:
            final_frame['atoms']['xyz'] = optimized_frame['atoms']['xyz']

        final_frame['bonds'] = concat_blocks(all_bonds)
        final_frame['angles'] = concat_blocks(all_angles)
        final_frame['dihedrals'] = concat_blocks(all_dihedrals)

        return final_frame
