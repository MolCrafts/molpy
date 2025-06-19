import io
import re
from datetime import datetime
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd

import molpy as mp

from .base import DataReader, DataWriter


class LammpsDataReader(DataReader):

    def __init__(self, path: str | Path, atom_style="full"):
        super().__init__(path)
        self.atom_style = atom_style
        self._file = None

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def read(self, frame: mp.Frame) -> mp.Frame:

        ff = mp.ForceField()
        if ff.n_atomstyles == 0:
            atom_style = ff.def_atomstyle(self.atom_style)
        else:
            atom_style = ff.atomstyles[0]
        if ff.n_bondstyles == 0:
            bond_style = None
        else:
            bond_style = ff.bondstyles[0]
        if ff.n_anglestyles == 0:
            angle_style = None
        else:
            angle_style = ff.anglestyles[0]
        if ff.n_dihedralstyles == 0:
            dihedral_style = None
        else:
            dihedral_style = ff.dihedralstyles[0]
        if ff.n_improperstyles == 0:
            improper_style = None
        else:
            improper_style = ff.improperstyles[0]

        with open(self._path, "r") as file:
            lines = list(filter(lambda line: line, map(LammpsDataReader.sanitizer, file)))

            box = {
                "xlo": 0.0,
                "xhi": 0.0,
                "ylo": 0.0,
                "yhi": 0.0,
                "zlo": 0.0,
                "zhi": 0.0,
            }

            props = {}
            masses = {}

            type_key = "type_id"

            line_iter = iter(lines)
            for line in line_iter:

            if line.endswith("atoms"):
                props["n_atoms"] = int(line.split()[0])

            elif line.endswith("bonds"):
                props["n_bonds"] = int(line.split()[0])

            elif line.endswith("angles"):
                props["n_angles"] = int(line.split()[0])

            elif line.endswith("dihedrals"):
                props["n_dihedrals"] = int(line.split()[0])

            elif line.endswith("impropers"):
                props["n_impropers"] = int(line.split()[0])

            elif line.endswith("atom types"):
                props["n_atomtypes"] = int(line.split()[0])

            elif line.endswith("bond types"):
                props["n_bondtypes"] = int(line.split()[0])

            elif line.endswith("angle types"):
                props["n_angletypes"] = int(line.split()[0])

            elif line.endswith("dihedral types"):
                props["n_dihedraltypes"] = int(line.split()[0])

            elif line.endswith("improper types"):
                props["n_impropertypes"] = int(line.split()[0])

            elif line.endswith("xlo xhi"):
                xlo, xhi = map(float, line.split()[:2])
                box["xlo"] = xlo
                box["xhi"] = xhi

            elif line.endswith("ylo yhi"):
                ylo, yhi = map(float, line.split()[:2])
                box["ylo"] = ylo
                box["yhi"] = yhi

            elif line.endswith("zlo zhi"):
                zlo, zhi = map(float, line.split()[:2])
                box["zlo"] = zlo
                box["zhi"] = zhi

            elif line.startswith("Masses"):
                mass_lines = list(islice(lines, props.get("n_atomtypes", 0)))
                for mass_line in mass_lines:
                    parts = mass_line.split()
                    if len(parts) >= 2:
                        masses[parts[0]] = float(parts[1])

            elif line.startswith("Atoms"):

                atom_lines = list(islice(lines, props.get("n_atoms", 0)))
                if not atom_lines:
                    continue
                    
                probe_line_len = len(atom_lines[0].split())
                header = []

                match self.atom_style:

                    case "full":
                        header = ["id", "molid", type_key, "charge", "x", "y", "z"]
                        if probe_line_len > 7:
                            if probe_line_len >= 10:
                                header.extend(["ix", "iy", "iz"])
                            if "#" in atom_lines[0].split():
                                content, sep, comment = atom_lines[0].partition("#")
                                if comment:
                                    header.extend(["seq", "name"])
                
                if header:
                    atom_table = pd.read_csv(
                        io.BytesIO("\n".join(atom_lines).encode()),
                        names=header,
                        delimiter=" ",
                    )
                    if masses:
                        atom_table["mass"] = atom_table[type_key].map(
                            lambda t: masses.get(str(t), 0.0)
                        )

                    frame["atoms"] = atom_table.to_xarray()

            elif line.startswith("Bonds"):
                bond_lines = list(islice(lines, props["n_bonds"]))
                bond_table = pd.read_csv(
                    io.BytesIO("\n".join(bond_lines).encode()),
                    names=["id", type_key, "i", "j"],
                    delimiter=" ",
                )
                frame["bonds"] = bond_table.to_xarray()

            elif line.startswith("Angles"):
                angle_lines = list(islice(lines, props["n_angles"]))
                angle_table = pd.read_csv(
                    io.BytesIO("\n".join(angle_lines).encode()),
                    names=["id", type_key, "i", "j", "k"],
                    delimiter=" ",
                )
                frame["angles"] = angle_table.to_xarray()

            elif line.startswith("Dihedrals"):
                dihedral_lines = list(islice(lines, props["n_dihedrals"]))
                dihedral_table = pd.read_csv(
                    io.BytesIO("\n".join(dihedral_lines).encode()),
                    names=["id", type_key, "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["dihedrals"] = dihedral_table.to_xarray()

            elif line.startswith("Impropers"):
                improper_lines = list(islice(lines, props["n_impropers"]))
                improper_table = pd.read_csv(
                    io.BytesIO("\n".join(improper_lines).encode()),
                    names=["id", type_key, "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["impropers"] = improper_table.to_xarray()

            elif line.startswith("Atom Type Labels"):
                type_key = "type"
                label_lines = list(islice(lines, props.get("n_atomtypes", 0)))
                for label_line in label_lines:
                    parts = label_line.split()
                    if len(parts) >= 2:
                        atom_type = int(parts[0])
                        if atom_type not in atom_style:
                            at = atom_style.def_type(
                                parts[1], kw_params={"id": atom_type}
                            )
                        else:
                            atom_style[atom_type].name = parts[1]

            elif line.startswith("Bond Type Labels"):
                if not bond_style:
                    continue
                label_lines = list(islice(lines, props.get("n_bondtypes", 0)))
                for label_line in label_lines:
                    parts = label_line.split()
                    if len(parts) >= 2:
                        bond_type = int(parts[0])
                        if bond_type not in bond_style:
                            bt = bond_style.def_type(
                                parts[1], kw_params={"id": bond_type}
                            )
                        else:
                            bond_style[bond_type].name = parts[1]

            elif line.startswith("Angle Type Labels"):
                if not angle_style:
                    continue
                label_lines = list(islice(lines, props.get("n_angletypes", 0)))
                for label_line in label_lines:
                    parts = label_line.split()
                    if len(parts) >= 2:
                        angle_type = int(parts[0])
                        if angle_type not in angle_style:
                            at = angle_style.def_type(
                                parts[1], kw_params={"id": angle_type}
                            )
                        else:
                            angle_style[angle_type].name = parts[1]

            elif line.startswith("Dihedral Type Labels"):
                if not dihedral_style:
                    continue
                label_lines = list(islice(lines, props.get("n_dihedraltypes", 0)))
                for label_line in label_lines:
                    parts = label_line.split()
                    if len(parts) >= 2:
                        dihedral_type = int(parts[0])
                        if dihedral_type not in dihedral_style:
                            dt = dihedral_style.def_type(
                                parts[1], kw_params={"id": dihedral_type}
                            )
                        else:
                            dihedral_style[dihedral_type].name = parts[1]

            elif line.startswith("Improper Type Labels"):
                if not improper_style:
                    continue
                label_lines = list(islice(lines, props.get("n_impropertypes", 0)))
                for label_line in label_lines:
                    parts = label_line.split()
                    if len(parts) >= 2:
                        improper_type = int(parts[0])
                        if improper_type not in improper_style:
                            it = improper_style.def_type(
                                parts[1], kw_params={"id": improper_type}
                            )
                        else:
                            improper_style[improper_type].name = parts[1]

        per_atom_mass = np.zeros(props.get("n_atoms", 0))
        for t, m in masses.items():
            atomtype = atom_style.get_by(lambda atom: atom.name == str(t))
            if not atomtype:
                atomtype = atom_style.def_type(str(t), kw_params={"id": t})
            atomtype["mass"] = m

            if "atoms" in frame:
                per_atom_mass[frame["atoms"][type_key] == t] = m  # todo: type mapping
        
        if "atoms" in frame:
            atoms_ds = frame["atoms"].copy()
            atoms_ds["mass"] = ("index", per_atom_mass)
            frame["atoms"] = atoms_ds

        box = mp.Box(
            np.diag(
                [
                    box["xhi"] - box["xlo"],
                    box["yhi"] - box["ylo"],
                    box["zhi"] - box["zlo"],
                ]
            )
        )
        frame.box = box

        return frame


class LammpsDataWriter(DataWriter):

    def __init__(self, path: Path, atom_style: str = "full", ):
        super().__init__(path)
        self._atom_style = atom_style

    def write(self, frame):

        ff = getattr(frame, 'forcefield', None)

        n_atoms = frame["atoms"].sizes.get("index", 0)
        n_bonds = frame["bonds"].sizes.get("index", 0) if "bonds" in frame else 0
        n_angles = frame["angles"].sizes.get("index", 0) if "angles" in frame else 0
        n_dihedrals = (
            frame["dihedrals"].sizes.get("index", 0) if "dihedrals" in frame else 0
        )

        with open(self._path, "w") as f:

            f.write(f"# generated by molpy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"{n_atoms} atoms\n")
            
            if ff and hasattr(ff, 'n_atomtypes'):
                f.write(f"{ff.n_atomtypes} atom types\n")
            else:
                # Fallback: count unique atom types from frame
                if "atoms" in frame and "type" in frame["atoms"]:
                    n_atom_types = len(np.unique(frame["atoms"]["type"].values))
                    f.write(f"{n_atom_types} atom types\n")
                else:
                    f.write("1 atom types\n")
                    
            if "bonds" in frame:
                f.write(f"{n_bonds} bonds\n")
                if ff and hasattr(ff, 'n_bondtypes'):
                    f.write(f"{ff.n_bondtypes} bond types\n")
                else:
                    n_bond_types = len(np.unique(frame["bonds"]["type"].values)) if "type" in frame["bonds"] else 1
                    f.write(f"{n_bond_types} bond types\n")
                    
            if "angles" in frame:
                f.write(f"{n_angles} angles\n")
                if ff and hasattr(ff, 'n_angletypes'):
                    f.write(f"{ff.n_angletypes} angle types\n")
                else:
                    n_angle_types = len(np.unique(frame["angles"]["type"].values)) if "type" in frame["angles"] else 1
                    f.write(f"{n_angle_types} angle types\n")
                    
            if "dihedrals" in frame:
                f.write(f"{n_dihedrals} dihedrals\n")
                if ff and hasattr(ff, 'n_dihedraltypes'):
                    f.write(f"{ff.n_dihedraltypes} dihedral types\n\n")
                else:
                    n_dihedral_types = len(np.unique(frame["dihedrals"]["type"].values)) if "type" in frame["dihedrals"] else 1
                    f.write(f"{n_dihedral_types} dihedral types\n\n")

            box = getattr(frame, 'box', None)
            if box and hasattr(box, 'xlo'):
                xlo = box.xlo
                xhi = box.xhi
                ylo = box.ylo
                yhi = box.yhi
                zlo = box.zlo
                zhi = box.zhi
            else:
                # Fallback: use atom coordinates to estimate box
                if "atoms" in frame and "x" in frame["atoms"] and "y" in frame["atoms"] and "z" in frame["atoms"]:
                    coords = np.column_stack([
                        frame["atoms"]["x"].values,
                        frame["atoms"]["y"].values, 
                        frame["atoms"]["z"].values
                    ])
                    xlo, ylo, zlo = coords.min(axis=0) - 1.0
                    xhi, yhi, zhi = coords.max(axis=0) + 1.0
                else:
                    xlo = ylo = zlo = 0.0
                    xhi = yhi = zhi = 10.0

            f.write(f"{xlo:.6f} {xhi:.6f} xlo xhi\n")
            f.write(f"{ylo:.6f} {yhi:.6f} ylo yhi\n")
            f.write(f"{zlo:.6f} {zhi:.6f} zlo zhi\n")

            type_key = "type"

            if ff and hasattr(ff, 'atomstyles') and ff.atomstyles:

                found_type_label_in_atom = "type" in frame["atoms"]
                type_key = "type" if found_type_label_in_atom else "type_id"

                if type_key == "type":
                    f.write(f"\nAtom Type Labels\n\n")
                    for i, atomtype in enumerate(ff.get_atomtypes(), 1):
                        f.write(f"{i} {atomtype.name}\n")

                if hasattr(ff, 'bondstyles') and ff.bondstyles:
                
                    f.write(f"\nBond Type Labels\n\n")
                    for i, bondtype in enumerate(ff.get_bondtypes(), 1):
                        f.write(f"{i} {bondtype.name}\n")

                if hasattr(ff, 'anglestyles') and ff.anglestyles:
                
                    f.write(f"\nAngle Type Labels\n\n")
                    for i, angletype in enumerate(ff.get_angletypes(), 1):
                        f.write(f"{i} {angletype.name}\n")

                if hasattr(ff, 'dihedralstyles') and ff.dihedralstyles:
                
                    f.write(f"\nDihedral Type Labels\n\n")
                    for i, dihedraltype in enumerate(ff.get_dihedraltypes(), 1):
                        f.write(f"{i} {dihedraltype.name}\n")

            f.write(f"\nMasses\n\n")
            masses = {}
            try:
                if ff and hasattr(ff, 'n_atomtypes') and ff.n_atomtypes:
                    for atomtype in ff.get_atomtypes():
                        masses[atomtype.name] = atomtype["mass"]
                else:
                    raise KeyError("n_atomtypes")
            except (KeyError, AttributeError):
                masses = {}
                if "atoms" in frame and "type" in frame["atoms"]:
                    unique_type, unique_idx = np.unique(
                        frame["atoms"]["type"].to_numpy(), return_index=True
                    )
                    if "mass" in frame["atoms"]:
                        mass_arr = frame["atoms"]["mass"].to_numpy()
                        for i, t in zip(unique_idx, unique_type):
                            masses[t] = mass_arr[i]
                    else:
                        # Default masses
                        for t in unique_type:
                            masses[t] = 1.0

            for i, m in masses.items():
                f.write(f"{i} {m:.3f}\n")

            f.write(f"\nAtoms\n\n")
            if "atoms" not in frame:
                return
                
            atoms = frame["atoms"]
            if "id" not in atoms:
                # Create sequential IDs starting from 1
                atoms = atoms.assign(id=np.arange(1, len(atoms) + 1))
                
            match self._atom_style:
                case "full":
                    # Efficiently write atom lines using vectorized access
                    # Ensure required columns exist
                    required_cols = ["id", "molid", "type", "q", "x", "y", "z"]
                    missing_cols = [col for col in required_cols if col not in atoms]
                    
                    if missing_cols:
                        # Create default values for missing columns
                        if "molid" not in atoms:
                            atoms = atoms.assign(molid=1)
                        if "q" not in atoms:
                            atoms = atoms.assign(q=0.0)
                        if "x" not in atoms and "xyz" in atoms:
                            xyz = atoms["xyz"].values
                            atoms = atoms.assign(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2])
                        elif "x" not in atoms:
                            atoms = atoms.assign(x=0.0, y=0.0, z=0.0)
                    
                    # Write atoms data
                    for i in range(len(atoms["id"])):
                        atom_id = atoms["id"].values[i]
                        molid = atoms["molid"].values[i]
                        atom_type = atoms["type"].values[i]
                        charge = atoms["q"].values[i]
                        x = atoms["x"].values[i]  
                        y = atoms["y"].values[i]
                        z = atoms["z"].values[i]
                        
                        f.write(f"{atom_id} {molid} {atom_type} {charge:.6f} {x:.6f} {y:.6f} {z:.6f}\n")

            if "bonds" in frame:
                bonds = frame["bonds"]
                if "id" not in bonds:
                    bonds = bonds.assign(id=np.arange(1, len(bonds) + 1))
                f.write(f"\nBonds\n\n")
                for i in range(len(bonds["id"])):
                    bond_id = bonds["id"].values[i]
                    bond_type = bonds[type_key].values[i]
                    atom_i = bonds["i"].values[i]
                    atom_j = bonds["j"].values[i]
                    f.write(f"{bond_id} {bond_type} {atom_i} {atom_j}\n")

            if "angles" in frame:
                angles = frame["angles"]
                if "id" not in angles:
                    angles = angles.assign(id=np.arange(1, len(angles) + 1))
                f.write(f"\nAngles\n\n")
                for i in range(len(angles["id"])):
                    angle_id = angles["id"].values[i]
                    angle_type = angles[type_key].values[i]
                    atom_i = angles["i"].values[i]
                    atom_j = angles["j"].values[i]
                    atom_k = angles["k"].values[i]
                    f.write(f"{angle_id} {angle_type} {atom_i} {atom_j} {atom_k}\n")

            if "dihedrals" in frame:
                dihedrals = frame["dihedrals"]
                if "id" not in dihedrals:
                    dihedrals = dihedrals.assign(id=np.arange(1, len(dihedrals) + 1))
                f.write(f"\nDihedrals\n\n")
                for i in range(len(dihedrals["id"])):
                    dihedral_id = dihedrals["id"].values[i]
                    dihedral_type = dihedrals[type_key].values[i]
                    atom_i = dihedrals["i"].values[i]
                    atom_j = dihedrals["j"].values[i]
                    atom_k = dihedrals["k"].values[i]
                    atom_l = dihedrals["l"].values[i]
                    f.write(f"{dihedral_id} {dihedral_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")

            if "impropers" in frame:
                impropers = frame["impropers"]
                if "id" not in impropers:
                    impropers = impropers.assign(id=np.arange(1, len(impropers) + 1))
                f.write(f"\nImpropers\n\n")
                for i in range(len(impropers["id"])):
                    improper_id = impropers["id"].values[i]
                    improper_type = impropers[type_key].values[i]
                    atom_i = impropers["i"].values[i]
                    atom_j = impropers["j"].values[i]
                    atom_k = impropers["k"].values[i]
                    atom_l = impropers["l"].values[i]
                    f.write(f"{improper_id} {improper_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")


class LammpsMoleculeReader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._path = Path(file)
        self.style = "full"
        with open(self._path, "r") as f:
            self.lines = list(filter(lambda line: line, map(LammpsDataReader.sanitizer, f)))

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def read(self, frame: mp.Frame) -> mp.Frame:

        props = {}
        line_iter = iter(self.lines)

        for line in line_iter:

            if line.endswith("atoms"):
                props["n_atoms"] = int(line.split()[0])

            elif line.endswith("bonds"):
                props["n_bonds"] = int(line.split()[0])

            elif line.endswith("angles"):
                props["n_angles"] = int(line.split()[0])

            elif line.endswith("dihedrals"):
                props["n_dihedrals"] = int(line.split()[0])

            elif line.endswith("impropers"):
                props["n_impropers"] = int(line.split()[0])

            elif line.startswith("Coords"):
                header = ["id", "x", "y", "z"]
                atom_lines = []
                for _ in range(props.get("n_atoms", 0)):
                    try:
                        atom_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if atom_lines:
                    atom_table = pd.read_csv(
                        io.BytesIO("\n".join(atom_lines).encode()),
                        names=header,
                        delimiter=" ",
                    )
                    frame["atoms"] = atom_table.to_xarray()

            elif line.startswith("Types"):
                header = ["id", "type"]
                atomtype_lines = []
                for _ in range(props.get("n_atoms", 0)):
                    try:
                        atomtype_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if atomtype_lines:
                    atomtype_table = pd.read_csv(
                        io.BytesIO("\n".join(atomtype_lines).encode()),
                        names=header,
                        delimiter=" ",
                    )
                    # join atom table and type table
                    if "atoms" in frame:
                        atoms_df = frame["atoms"].to_pandas()
                        types_df = atomtype_table
                        merged_df = atoms_df.merge(types_df, on="id", how="left")
                        frame["atoms"] = merged_df.to_xarray()

            elif line.startswith("Charges"):
                header = ["id", "charge"]
                charge_lines = []
                for _ in range(props.get("n_atoms", 0)):
                    try:
                        charge_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if charge_lines:
                    charge_table = pd.read_csv(
                        io.BytesIO("\n".join(charge_lines).encode()),
                        names=header,
                        delimiter=" ",
                    )
                    if "atoms" in frame:
                        atoms_df = frame["atoms"].to_pandas()
                        charges_df = charge_table
                        merged_df = atoms_df.merge(charges_df, on="id", how="left")
                        frame["atoms"] = merged_df.to_xarray()

            elif line.startswith("Molecules"):
                header = ["id", "molid"]
                molid_lines = []
                for _ in range(props.get("n_atoms", 0)):
                    try:
                        molid_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if molid_lines:
                    molid_table = pd.read_csv(
                        io.BytesIO("\n".join(molid_lines).encode()),
                        names=header,
                        delimiter=" ",
                    )
                    if "atoms" in frame:
                        atoms_df = frame["atoms"].to_pandas()
                        molids_df = molid_table
                        merged_df = atoms_df.merge(molids_df, on="id", how="left")
                        frame["atoms"] = merged_df.to_xarray()

            elif line.startswith("Bonds"):
                bond_lines = []
                for _ in range(props.get("n_bonds", 0)):
                    try:
                        bond_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if bond_lines:
                    bond_table = pd.read_csv(
                        io.BytesIO("\n".join(bond_lines).encode()),
                        names=["id", "type", "i", "j"],
                        delimiter=" ",
                    )
                    frame["bonds"] = bond_table.to_xarray()

            elif line.startswith("Angles"):
                angle_lines = []
                for _ in range(props.get("n_angles", 0)):
                    try:
                        angle_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if angle_lines:
                    angle_table = pd.read_csv(
                        io.BytesIO("\n".join(angle_lines).encode()),
                        names=["id", "type", "i", "j", "k"],
                        delimiter=" ",
                    )
                    frame["angles"] = angle_table.to_xarray()

            elif line.startswith("Dihedrals"):
                dihedral_lines = []
                for _ in range(props.get("n_dihedrals", 0)):
                    try:
                        dihedral_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if dihedral_lines:
                    dihedral_table = pd.read_csv(
                        io.BytesIO("\n".join(dihedral_lines).encode()),
                        names=["id", "type", "i", "j", "k", "l"],
                        delimiter=" ",
                    )
                    frame["dihedrals"] = dihedral_table.to_xarray()

            elif line.startswith("Impropers"):
                improper_lines = []
                for _ in range(props.get("n_impropers", 0)):
                    try:
                        improper_lines.append(next(line_iter))
                    except StopIteration:
                        break
                        
                if improper_lines:
                    improper_table = pd.read_csv(
                        io.BytesIO("\n".join(improper_lines).encode()),
                        names=["id", "type", "i", "j", "k", "l"],
                        delimiter=" ",
                    )
                    frame["impropers"] = improper_table.to_xarray()
                    
        return frame


class LammpsMoleculeWriter:

    def __init__(self, file: str | Path, atom_style: str = "full"):
        self._path = Path(file)
        self._atom_style = atom_style

    def write(self, frame: mp.Frame):

        with open(self._path, "w") as f:

            f.write(f"# generated by molpy\n\n")

            n_atoms = len(frame["atoms"]) if "atoms" in frame else 0
            f.write(f"{n_atoms} atoms\n")
            n_bonds = len(frame["bonds"]) if "bonds" in frame else 0
            f.write(f"{n_bonds} bonds\n")
            n_angles = len(frame["angles"]) if "angles" in frame else 0
            f.write(f"{n_angles} angles\n")
            n_dihedrals = len(frame["dihedrals"]) if "dihedrals" in frame else 0
            f.write(f"{n_dihedrals} dihedrals\n\n")

            if "atoms" in frame:
                atoms = frame["atoms"]
                
                # Convert to pandas for easier iteration
                if hasattr(atoms, 'to_pandas'):
                    atoms_df = atoms.to_pandas()
                else:
                    atoms_df = atoms
                
                # Write coordinates
                f.write(f"\nCoords\n\n")
                for i, row in atoms_df.iterrows():
                    if "x" in row and "y" in row and "z" in row:
                        f.write(f"{i} {row['x']:.6f} {row['y']:.6f} {row['z']:.6f}\n")
                    elif "xyz" in row:
                        xyz = row["xyz"]
                        f.write(f"{i} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}\n")

                # Write types
                if "type" in atoms_df.columns:
                    f.write(f"\nTypes\n\n")
                    for i, row in atoms_df.iterrows():
                        f.write(f"{i} {row['type']}\n")

                # Write charges
                if "q" in atoms_df.columns or "charge" in atoms_df.columns:
                    f.write(f"\nCharges\n\n")
                    charge_col = "q" if "q" in atoms_df.columns else "charge"
                    for i, row in atoms_df.iterrows():
                        f.write(f"{i} {float(row[charge_col]):.6f}\n")

                # Write molecule IDs
                if "molid" in atoms_df.columns:
                    f.write(f"\nMolecules\n\n")
                    for i, row in atoms_df.iterrows():
                        f.write(f"{i} {int(row['molid'])}\n")

            if "bonds" in frame:
                bonds = frame["bonds"]
                if hasattr(bonds, 'to_pandas'):
                    bonds_df = bonds.to_pandas()
                else:
                    bonds_df = bonds
                    
                f.write(f"\nBonds\n\n")
                for i, row in bonds_df.iterrows():
                    bond_type = row.get('type', 1)
                    atom_i = int(row['i'])
                    atom_j = int(row['j'])
                    f.write(f"{i} {bond_type} {atom_i} {atom_j}\n")

            if "angles" in frame:
                angles = frame["angles"]
                if hasattr(angles, 'to_pandas'):
                    angles_df = angles.to_pandas()
                else:
                    angles_df = angles
                    
                f.write(f"\nAngles\n\n")
                for i, row in angles_df.iterrows():
                    angle_type = row.get('type', 1)
                    atom_i = int(row['i'])
                    atom_j = int(row['j'])
                    atom_k = int(row['k'])
                    f.write(f"{i} {angle_type} {atom_i} {atom_j} {atom_k}\n")

            if "dihedrals" in frame:
                dihedrals = frame["dihedrals"]
                if hasattr(dihedrals, 'to_pandas'):
                    dihedrals_df = dihedrals.to_pandas()
                else:
                    dihedrals_df = dihedrals
                    
                f.write(f"\nDihedrals\n\n")
                for i, row in dihedrals_df.iterrows():
                    dihedral_type = row.get('type', 1)
                    atom_i = int(row['i'])
                    atom_j = int(row['j'])
                    atom_k = int(row['k'])
                    atom_l = int(row['l'])
                    f.write(f"{i} {dihedral_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")

            if "impropers" in frame:
                impropers = frame["impropers"]
                if hasattr(impropers, 'to_pandas'):
                    impropers_df = impropers.to_pandas()
                else:
                    impropers_df = impropers
                    
                f.write(f"\nImpropers\n\n")
                for i, row in impropers_df.iterrows():
                    improper_type = row.get('type', 1)
                    atom_i = int(row['i'])
                    atom_j = int(row['j'])
                    atom_k = int(row['k'])
                    atom_l = int(row['l'])
                    f.write(f"{i} {improper_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")
