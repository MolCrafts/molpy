from pathlib import Path
import shutil
import molpy as mp
import xarray as xr
import molpy.pack as mpk
import subprocess
import tempfile
import numpy as np
from .base import Packer


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

    def pack(self, targets=None, max_steps: int = 1000, seed: int | None = None) -> mp.Frame:
        if targets is None:
            targets = self.targets
        if seed is None:
            seed = 4628
        self.generate_input(targets, max_steps, seed)
        subprocess.run(
            f"{self.cmd} < .packmol.inp > .packmol.out",
            shell=True
        )
        optimized_frame = mp.io.read_pdb(Path(".optimized.pdb"))
        self.remove_input()
        atoms = []
        bonds = []
        angles = []
        dihedrals = []
        n_atoms = np.cumsum(
            [
                len(target.frame["atoms"])
                for target in targets
                for _ in range(target.number)
            ]
        )
        frame = mp.Frame()
        n_atoms = np.concatenate([[0], n_atoms])
        n_struct_added = 0
        n_atom_types = 0
        last_target = None  # Track the last target for final checks
        
        for target in targets:
            last_target = target
            last_atoms = None  # Track atoms for type counting
            
            for i in range(target.number):
                a = target.frame["atoms"].copy()
                if "molid" not in a:
                    a["molid"] = n_struct_added + 1
                a["id"] = a["id"] + n_atoms[n_struct_added + i]
                a["molid"] = a["molid"] + n_struct_added + i
                if "type" in a:
                    a["type"] = a["type"] + n_atom_types
                atoms.append(a)
                last_atoms = a  # Keep reference to last atoms for type counting
                
            if "bonds" in target.frame:
                for i in range(target.number):
                    b = target.frame["bonds"].copy()
                    b["id"] = b["id"] + n_atoms[n_struct_added + i]
                    b[["i", "j"]] = b[["i", "j"]] + n_atoms[n_struct_added + i]
                    bonds.append(b)
            if "angles" in target.frame:
                for i in range(target.number):
                    an = target.frame["angles"].copy()
                    an["id"] = an["id"] + n_atoms[n_struct_added + i]
                    an[["i", "j", "k"]] = (
                        an[["i", "j", "k"]] + n_atoms[n_struct_added + i]
                    )
                    angles.append(an)
            if "dihedrals" in target.frame:
                for i in range(target.number):
                    d = target.frame["dihedrals"].copy()
                    d["id"] = d["id"] + n_atoms[n_struct_added + i]
                    d[["i", "j", "k", "l"]] = (
                        d[["i", "j", "k", "l"]] + n_atoms[n_struct_added + i]
                    )
                    dihedrals.append(d)
            
            # Update atom types counter using the last atoms processed
            if last_atoms is not None and "type" in last_atoms:
                n_atom_types += last_atoms["type"].max()  # assume start with 1 always

            n_struct_added += target.number
            
        _atoms = xr.concat(atoms, dim="index")
        _atoms[["x", "y", "z"]] = optimized_frame["atoms"][["x", "y", "z"]]
        frame["atoms"] = _atoms
        
        ## WARNING: assume all targets has/dont have bonds, angles, dihedrals
        if last_target and "bonds" in last_target.frame:
            frame["bonds"] = xr.concat(bonds, dim="index")
        if last_target and "angles" in last_target.frame:
            frame["angles"] = xr.concat(angles, dim="index")
        if last_target and "dihedrals" in last_target.frame:
            frame["dihedrals"] = xr.concat(dihedrals, dim="index")
        return frame