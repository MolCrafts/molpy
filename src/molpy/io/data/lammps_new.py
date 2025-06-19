import io
import re
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import Dict, Any, List, Iterator

import numpy as np
import pandas as pd

import molpy as mp

from .base import DataReader, DataWriter


class LammpsDataReader(DataReader):

    def __init__(self, path: str | Path, atom_style="full"):
        super().__init__(path)
        self.atom_style = atom_style

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def read(self, frame: mp.Frame) -> mp.Frame:
        # Get ForceField instance
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

        # Read and sanitize lines
        lines = [self.sanitizer(line) for line in self.read_lines()]
        line_iter = iter(lines)

        box = {
            "xlo": 0.0, "xhi": 0.0,
            "ylo": 0.0, "yhi": 0.0,
            "zlo": 0.0, "zhi": 0.0,
        }

        props = {}
        masses = {}
        type_key = "type_id"

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
                box.update({"xlo": xlo, "xhi": xhi})
            elif line.endswith("ylo yhi"):
                ylo, yhi = map(float, line.split()[:2])
                box.update({"ylo": ylo, "yhi": yhi})
            elif line.endswith("zlo zhi"):
                zlo, zhi = map(float, line.split()[:2])
                box.update({"zlo": zlo, "zhi": zhi})
            elif line.startswith("Masses"):
                self._read_masses(line_iter, props, masses)
            elif line.startswith("Atoms"):
                self._read_atoms(line_iter, props, frame, type_key, masses)
            elif line.startswith("Bonds"):
                self._read_bonds(line_iter, props, frame, type_key)
            elif line.startswith("Angles"):
                self._read_angles(line_iter, props, frame, type_key)
            elif line.startswith("Dihedrals"):
                self._read_dihedrals(line_iter, props, frame, type_key)
            elif line.startswith("Impropers"):
                self._read_impropers(line_iter, props, frame, type_key)
            elif line.startswith("Atom Type Labels"):
                type_key = "type"
                self._read_atom_type_labels(line_iter, props, atom_style)
            elif line.startswith("Bond Type Labels"):
                self._read_bond_type_labels(line_iter, props, bond_style)
            elif line.startswith("Angle Type Labels"):
                self._read_angle_type_labels(line_iter, props, angle_style)
            elif line.startswith("Dihedral Type Labels"):
                self._read_dihedral_type_labels(line_iter, props, dihedral_style)
            elif line.startswith("Improper Type Labels"):
                self._read_improper_type_labels(line_iter, props, improper_style)

        # Set masses and box
        self._set_masses(frame, masses, atom_style, type_key, props)
        self._set_box(frame, box)

        return frame

    def _read_masses(self, line_iter: Iterator[str], props: Dict[str, Any], masses: Dict[str, float]):
        """Read masses section."""
        mass_lines = list(islice(line_iter, props.get("n_atomtypes", 0)))
        for mass_line in mass_lines:
            parts = mass_line.split()
            if len(parts) >= 2:
                masses[parts[0]] = float(parts[1])

    def _read_atoms(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame, 
                   type_key: str, masses: Dict[str, float]):
        """Read atoms section."""
        atom_lines = list(islice(line_iter, props.get("n_atoms", 0)))
        if not atom_lines:
            return

        probe_line_len = len(atom_lines[0].split())
        header = self._get_atom_header(type_key, probe_line_len, atom_lines[0])
        
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

    def _get_atom_header(self, type_key: str, probe_line_len: int, first_line: str) -> List[str]:
        """Get header for atoms based on atom_style."""
        if self.atom_style == "full":
            header = ["id", "molid", type_key, "charge", "x", "y", "z"]
            if probe_line_len > 7:
                if probe_line_len >= 10:
                    header.extend(["ix", "iy", "iz"])
                if "#" in first_line:
                    content, sep, comment = first_line.partition("#")
                    if comment:
                        header.extend(["seq", "name"])
            return header
        return []

    def _read_bonds(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame, type_key: str):
        """Read bonds section."""
        bond_lines = list(islice(line_iter, props.get("n_bonds", 0)))
        if bond_lines:
            bond_table = pd.read_csv(
                io.BytesIO("\n".join(bond_lines).encode()),
                names=["id", type_key, "i", "j"],
                delimiter=" ",
            )
            frame["bonds"] = bond_table.to_xarray()

    def _read_angles(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame, type_key: str):
        """Read angles section."""
        angle_lines = list(islice(line_iter, props.get("n_angles", 0)))
        if angle_lines:
            angle_table = pd.read_csv(
                io.BytesIO("\n".join(angle_lines).encode()),
                names=["id", type_key, "i", "j", "k"],
                delimiter=" ",
            )
            frame["angles"] = angle_table.to_xarray()

    def _read_dihedrals(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame, type_key: str):
        """Read dihedrals section."""
        dihedral_lines = list(islice(line_iter, props.get("n_dihedrals", 0)))
        if dihedral_lines:
            dihedral_table = pd.read_csv(
                io.BytesIO("\n".join(dihedral_lines).encode()),
                names=["id", type_key, "i", "j", "k", "l"],
                delimiter=" ",
            )
            frame["dihedrals"] = dihedral_table.to_xarray()

    def _read_impropers(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame, type_key: str):
        """Read impropers section."""
        improper_lines = list(islice(line_iter, props.get("n_impropers", 0)))
        if improper_lines:
            improper_table = pd.read_csv(
                io.BytesIO("\n".join(improper_lines).encode()),
                names=["id", type_key, "i", "j", "k", "l"],
                delimiter=" ",
            )
            frame["impropers"] = improper_table.to_xarray()

    def _read_atom_type_labels(self, line_iter: Iterator[str], props: Dict[str, Any], atom_style):
        """Read atom type labels section."""
        if not atom_style:
            return
        label_lines = list(islice(line_iter, props.get("n_atomtypes", 0)))
        for label_line in label_lines:
            parts = label_line.split()
            if len(parts) >= 2:
                atom_type = int(parts[0])
                try:
                    if atom_type not in atom_style:
                        atom_style.def_type(parts[1], kw_params={"id": atom_type})
                    else:
                        atom_style[atom_type].name = parts[1]
                except Exception:
                    # Handle ForceField API issues gracefully
                    pass

    def _read_bond_type_labels(self, line_iter: Iterator[str], props: Dict[str, Any], bond_style):
        """Read bond type labels section."""
        if not bond_style:
            return
        label_lines = list(islice(line_iter, props.get("n_bondtypes", 0)))
        for label_line in label_lines:
            parts = label_line.split()
            if len(parts) >= 2:
                bond_type = int(parts[0])
                try:
                    if bond_type not in bond_style:
                        bond_style.def_type(parts[1], kw_params={"id": bond_type})
                    else:
                        bond_style[bond_type].name = parts[1]
                except Exception:
                    pass

    def _read_angle_type_labels(self, line_iter: Iterator[str], props: Dict[str, Any], angle_style):
        """Read angle type labels section."""
        if not angle_style:
            return
        label_lines = list(islice(line_iter, props.get("n_angletypes", 0)))
        for label_line in label_lines:
            parts = label_line.split()
            if len(parts) >= 2:
                angle_type = int(parts[0])
                try:
                    if angle_type not in angle_style:
                        angle_style.def_type(parts[1], kw_params={"id": angle_type})
                    else:
                        angle_style[angle_type].name = parts[1]
                except Exception:
                    pass

    def _read_dihedral_type_labels(self, line_iter: Iterator[str], props: Dict[str, Any], dihedral_style):
        """Read dihedral type labels section."""
        if not dihedral_style:
            return
        label_lines = list(islice(line_iter, props.get("n_dihedraltypes", 0)))
        for label_line in label_lines:
            parts = label_line.split()
            if len(parts) >= 2:
                dihedral_type = int(parts[0])
                try:
                    if dihedral_type not in dihedral_style:
                        dihedral_style.def_type(parts[1], kw_params={"id": dihedral_type})
                    else:
                        dihedral_style[dihedral_type].name = parts[1]
                except Exception:
                    pass

    def _read_improper_type_labels(self, line_iter: Iterator[str], props: Dict[str, Any], improper_style):
        """Read improper type labels section."""
        if not improper_style:
            return
        label_lines = list(islice(line_iter, props.get("n_impropertypes", 0)))
        for label_line in label_lines:
            parts = label_line.split()
            if len(parts) >= 2:
                improper_type = int(parts[0])
                try:
                    if improper_type not in improper_style:
                        improper_style.def_type(parts[1], kw_params={"id": improper_type})
                    else:
                        improper_style[improper_type].name = parts[1]
                except Exception:
                    pass

    def _set_masses(self, frame: mp.Frame, masses: Dict[str, float], atom_style, type_key: str, props: Dict[str, Any]):
        """Set masses for atoms."""
        if "atoms" not in frame or not masses:
            return
            
        per_atom_mass = np.zeros(props.get("n_atoms", 0))
        for t, m in masses.items():
            try:
                atomtype = atom_style.get_by(lambda atom: atom.name == str(t))
                if not atomtype:
                    atomtype = atom_style.def_type(str(t), kw_params={"id": t})
                atomtype["mass"] = m
                per_atom_mass[frame["atoms"][type_key] == t] = m
            except Exception:
                # Handle ForceField API issues gracefully
                pass
        
        atoms_ds = frame["atoms"].copy()
        atoms_ds["mass"] = ("index", per_atom_mass)
        frame["atoms"] = atoms_ds

    def _set_box(self, frame: mp.Frame, box: Dict[str, float]):
        """Set simulation box."""
        frame.box = mp.Box(
            np.diag([
                box["xhi"] - box["xlo"],
                box["yhi"] - box["ylo"],
                box["zhi"] - box["zlo"],
            ])
        )


class LammpsDataWriter(DataWriter):

    def __init__(self, path: Path, atom_style: str = "full"):
        super().__init__(path)
        self._atom_style = atom_style

    def write(self, frame: mp.Frame):
        ff = getattr(frame, 'forcefield', None)

        n_atoms = frame["atoms"].sizes.get("index", 0)
        n_bonds = frame["bonds"].sizes.get("index", 0) if "bonds" in frame else 0
        n_angles = frame["angles"].sizes.get("index", 0) if "angles" in frame else 0
        n_dihedrals = frame["dihedrals"].sizes.get("index", 0) if "dihedrals" in frame else 0

        with open(self._path, "w") as f:
            # Header
            f.write(f"# generated by molpy at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Counts
            f.write(f"{n_atoms} atoms\n")
            self._write_type_counts(f, ff, frame)
            
            # Box dimensions
            self._write_box(f, frame)
            
            # Type labels
            self._write_type_labels(f, ff, frame)
            
            # Masses
            self._write_masses(f, ff, frame)
            
            # Atoms
            self._write_atoms(f, frame)
            
            # Bonds, Angles, Dihedrals, Impropers
            self._write_bonds(f, frame)
            self._write_angles(f, frame)
            self._write_dihedrals(f, frame)
            self._write_impropers(f, frame)

    def _write_type_counts(self, f, ff, frame):
        """Write atom/bond/angle/dihedral type counts."""
        if ff and hasattr(ff, 'n_atomtypes'):
            f.write(f"{ff.n_atomtypes} atom types\n")
        else:
            n_atom_types = len(np.unique(frame["atoms"]["type"].values)) if "atoms" in frame and "type" in frame["atoms"] else 1
            f.write(f"{n_atom_types} atom types\n")
            
        if "bonds" in frame:
            n_bonds = frame["bonds"].sizes.get("index", 0)
            f.write(f"{n_bonds} bonds\n")
            if ff and hasattr(ff, 'n_bondtypes'):
                f.write(f"{ff.n_bondtypes} bond types\n")
            else:
                n_bond_types = len(np.unique(frame["bonds"]["type"].values)) if "type" in frame["bonds"] else 1
                f.write(f"{n_bond_types} bond types\n")
                
        if "angles" in frame:
            n_angles = frame["angles"].sizes.get("index", 0)
            f.write(f"{n_angles} angles\n")
            if ff and hasattr(ff, 'n_angletypes'):
                f.write(f"{ff.n_angletypes} angle types\n")
            else:
                n_angle_types = len(np.unique(frame["angles"]["type"].values)) if "type" in frame["angles"] else 1
                f.write(f"{n_angle_types} angle types\n")
                
        if "dihedrals" in frame:
            n_dihedrals = frame["dihedrals"].sizes.get("index", 0)
            f.write(f"{n_dihedrals} dihedrals\n")
            if ff and hasattr(ff, 'n_dihedraltypes'):
                f.write(f"{ff.n_dihedraltypes} dihedral types\n")
            else:
                n_dihedral_types = len(np.unique(frame["dihedrals"]["type"].values)) if "type" in frame["dihedrals"] else 1
                f.write(f"{n_dihedral_types} dihedral types\n")

    def _write_box(self, f, frame):
        """Write box dimensions."""
        box = getattr(frame, 'box', None)
        if box and hasattr(box, 'xlo'):
            f.write(f"\n{box.xlo:.6f} {box.xhi:.6f} xlo xhi\n")
            f.write(f"{box.ylo:.6f} {box.yhi:.6f} ylo yhi\n")
            f.write(f"{box.zlo:.6f} {box.zhi:.6f} zlo zhi\n")
        else:
            # Fallback: estimate from coordinates
            if "atoms" in frame:
                coords = self._get_coords_array(frame["atoms"])
                if coords is not None:
                    mins = coords.min(axis=0) - 1.0
                    maxs = coords.max(axis=0) + 1.0
                    f.write(f"\n{mins[0]:.6f} {maxs[0]:.6f} xlo xhi\n")
                    f.write(f"{mins[1]:.6f} {maxs[1]:.6f} ylo yhi\n")
                    f.write(f"{mins[2]:.6f} {maxs[2]:.6f} zlo zhi\n")
                else:
                    f.write(f"\n0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi\n0.0 10.0 zlo zhi\n")

    def _get_coords_array(self, atoms):
        """Get coordinates array from atoms data."""
        try:
            if "x" in atoms and "y" in atoms and "z" in atoms:
                return np.column_stack([
                    atoms["x"].values,
                    atoms["y"].values,
                    atoms["z"].values
                ])
            elif "xyz" in atoms:
                return atoms["xyz"].values
        except Exception:
            pass
        return None

    def _write_type_labels(self, f, ff, frame):
        """Write type labels if available."""
        if ff and hasattr(ff, 'atomstyles') and ff.atomstyles:
            if "type" in frame["atoms"]:
                f.write(f"\nAtom Type Labels\n\n")
                try:
                    for i, atomtype in enumerate(ff.get_atomtypes(), 1):
                        f.write(f"{i} {atomtype.name}\n")
                except Exception:
                    pass

    def _write_masses(self, f, frame):
        """Write masses section."""
        f.write(f"\nMasses\n\n")
        masses = {}
        
        if "atoms" in frame and "type" in frame["atoms"]:
            unique_types = np.unique(frame["atoms"]["type"].values)
            if "mass" in frame["atoms"]:
                unique_type_indices = [
                    np.where(frame["atoms"]["type"].values == t)[0][0] 
                    for t in unique_types
                ]
                mass_arr = frame["atoms"]["mass"].values
                for i, t in zip(unique_type_indices, unique_types):
                    masses[t] = mass_arr[i]
            else:
                # Default masses
                for t in unique_types:
                    masses[t] = 1.0

        for t, m in masses.items():
            f.write(f"{t} {m:.3f}\n")

    def _write_atoms(self, f, frame):
        """Write atoms section."""
        if "atoms" not in frame:
            return
            
        f.write(f"\nAtoms\n\n")
        atoms = frame["atoms"]
        
        if self._atom_style == "full":
            self._write_atoms_full_style(f, atoms)

    def _write_atoms_full_style(self, f, atoms):
        """Write atoms in 'full' style."""
        required_cols = ["id", "molid", "type", "charge"]
        coords = self._get_coords_array(atoms)
        
        if coords is None:
            raise ValueError("No coordinate data found in atoms")
            
        # Ensure required columns exist with defaults
        atom_data = {}
        for col in required_cols:
            if col in atoms:
                atom_data[col] = atoms[col].values
            else:
                n_atoms = len(coords)
                if col == "id":
                    atom_data[col] = np.arange(1, n_atoms + 1)
                elif col == "molid":
                    atom_data[col] = np.ones(n_atoms, dtype=int)
                elif col == "type":
                    atom_data[col] = np.ones(n_atoms, dtype=int)
                elif col == "charge":
                    atom_data[col] = np.zeros(n_atoms)
        
        # Write atom lines
        for i in range(len(coords)):
            f.write(f"{int(atom_data['id'][i])} {int(atom_data['molid'][i])} "
                   f"{int(atom_data['type'][i])} {atom_data['charge'][i]:.6f} "
                   f"{coords[i][0]:.6f} {coords[i][1]:.6f} {coords[i][2]:.6f}\n")

    def _write_bonds(self, f, frame):
        """Write bonds section."""
        if "bonds" not in frame:
            return
            
        f.write(f"\nBonds\n\n")
        bonds = frame["bonds"]
        
        for i in range(bonds.sizes.get("index", 0)):
            bond_id = bonds["id"].values[i] if "id" in bonds else i + 1
            bond_type = bonds["type"].values[i] if "type" in bonds else 1
            atom_i = bonds["i"].values[i]
            atom_j = bonds["j"].values[i]
            f.write(f"{int(bond_id)} {int(bond_type)} {int(atom_i)} {int(atom_j)}\n")

    def _write_angles(self, f, frame):
        """Write angles section."""
        if "angles" not in frame:
            return
            
        f.write(f"\nAngles\n\n")
        angles = frame["angles"]
        
        for i in range(angles.sizes.get("index", 0)):
            angle_id = angles["id"].values[i] if "id" in angles else i + 1
            angle_type = angles["type"].values[i] if "type" in angles else 1
            atom_i = angles["i"].values[i]
            atom_j = angles["j"].values[i]
            atom_k = angles["k"].values[i]
            f.write(f"{int(angle_id)} {int(angle_type)} {int(atom_i)} {int(atom_j)} {int(atom_k)}\n")

    def _write_dihedrals(self, f, frame):
        """Write dihedrals section."""
        if "dihedrals" not in frame:
            return
            
        f.write(f"\nDihedrals\n\n")
        dihedrals = frame["dihedrals"]
        
        for i in range(dihedrals.sizes.get("index", 0)):
            dihedral_id = dihedrals["id"].values[i] if "id" in dihedrals else i + 1
            dihedral_type = dihedrals["type"].values[i] if "type" in dihedrals else 1
            atom_i = dihedrals["i"].values[i]
            atom_j = dihedrals["j"].values[i]
            atom_k = dihedrals["k"].values[i]
            atom_l = dihedrals["l"].values[i]
            f.write(f"{int(dihedral_id)} {int(dihedral_type)} {int(atom_i)} {int(atom_j)} {int(atom_k)} {int(atom_l)}\n")

    def _write_impropers(self, f, frame):
        """Write impropers section."""
        if "impropers" not in frame:
            return
            
        f.write(f"\nImpropers\n\n")
        impropers = frame["impropers"]
        
        for i in range(impropers.sizes.get("index", 0)):
            improper_id = impropers["id"].values[i] if "id" in impropers else i + 1
            improper_type = impropers["type"].values[i] if "type" in impropers else 1
            atom_i = impropers["i"].values[i]
            atom_j = impropers["j"].values[i]
            atom_k = impropers["k"].values[i]
            atom_l = impropers["l"].values[i]
            f.write(f"{int(improper_id)} {int(improper_type)} {int(atom_i)} {int(atom_j)} {int(atom_k)} {int(atom_l)}\n")


class LammpsMoleculeReader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self.style = "full"

    def read(self, frame: mp.Frame) -> mp.Frame:
        lines = self.read_lines()
        line_iter = iter(lines)
        props = {}

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
                self._read_coords(line_iter, props, frame)
            elif line.startswith("Types"):
                self._read_types(line_iter, props, frame)
            elif line.startswith("Charges"):
                self._read_charges(line_iter, props, frame)
            elif line.startswith("Molecules"):
                self._read_molecules(line_iter, props, frame)
            elif line.startswith("Bonds"):
                self._read_molecule_bonds(line_iter, props, frame)
            elif line.startswith("Angles"):
                self._read_molecule_angles(line_iter, props, frame)
            elif line.startswith("Dihedrals"):
                self._read_molecule_dihedrals(line_iter, props, frame)
            elif line.startswith("Impropers"):
                self._read_molecule_impropers(line_iter, props, frame)
                
        return frame

    def _read_coords(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read coordinates section."""
        coord_lines = list(islice(line_iter, props.get("n_atoms", 0)))
        if coord_lines:
            coord_table = pd.read_csv(
                io.BytesIO("\n".join(coord_lines).encode()),
                names=["id", "x", "y", "z"],
                delimiter=" ",
            )
            frame["atoms"] = coord_table.to_xarray()

    def _read_types(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read types section."""
        type_lines = list(islice(line_iter, props.get("n_atoms", 0)))
        if type_lines and "atoms" in frame:
            type_table = pd.read_csv(
                io.BytesIO("\n".join(type_lines).encode()),
                names=["id", "type"],
                delimiter=" ",
            )
            atoms_df = frame["atoms"].to_pandas()
            merged_df = atoms_df.merge(type_table, on="id", how="left")
            frame["atoms"] = merged_df.to_xarray()

    def _read_charges(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read charges section."""
        charge_lines = list(islice(line_iter, props.get("n_atoms", 0)))
        if charge_lines and "atoms" in frame:
            charge_table = pd.read_csv(
                io.BytesIO("\n".join(charge_lines).encode()),
                names=["id", "charge"],
                delimiter=" ",
            )
            atoms_df = frame["atoms"].to_pandas()
            merged_df = atoms_df.merge(charge_table, on="id", how="left")
            frame["atoms"] = merged_df.to_xarray()

    def _read_molecules(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read molecules section."""
        molid_lines = list(islice(line_iter, props.get("n_atoms", 0)))
        if molid_lines and "atoms" in frame:
            molid_table = pd.read_csv(
                io.BytesIO("\n".join(molid_lines).encode()),
                names=["id", "molid"],
                delimiter=" ",
            )
            atoms_df = frame["atoms"].to_pandas()
            merged_df = atoms_df.merge(molid_table, on="id", how="left")
            frame["atoms"] = merged_df.to_xarray()

    def _read_molecule_bonds(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read bonds section."""
        bond_lines = list(islice(line_iter, props.get("n_bonds", 0)))
        if bond_lines:
            bond_table = pd.read_csv(
                io.BytesIO("\n".join(bond_lines).encode()),
                names=["id", "type", "i", "j"],
                delimiter=" ",
            )
            frame["bonds"] = bond_table.to_xarray()

    def _read_molecule_angles(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read angles section."""
        angle_lines = list(islice(line_iter, props.get("n_angles", 0)))
        if angle_lines:
            angle_table = pd.read_csv(
                io.BytesIO("\n".join(angle_lines).encode()),
                names=["id", "type", "i", "j", "k"],
                delimiter=" ",
            )
            frame["angles"] = angle_table.to_xarray()

    def _read_molecule_dihedrals(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read dihedrals section."""
        dihedral_lines = list(islice(line_iter, props.get("n_dihedrals", 0)))
        if dihedral_lines:
            dihedral_table = pd.read_csv(
                io.BytesIO("\n".join(dihedral_lines).encode()),
                names=["id", "type", "i", "j", "k", "l"],
                delimiter=" ",
            )
            frame["dihedrals"] = dihedral_table.to_xarray()

    def _read_molecule_impropers(self, line_iter: Iterator[str], props: Dict[str, Any], frame: mp.Frame):
        """Read impropers section."""
        improper_lines = list(islice(line_iter, props.get("n_impropers", 0)))
        if improper_lines:
            improper_table = pd.read_csv(
                io.BytesIO("\n".join(improper_lines).encode()),
                names=["id", "type", "i", "j", "k", "l"],
                delimiter=" ",
            )
            frame["impropers"] = improper_table.to_xarray()


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
                self._write_molecule_atoms(f, frame["atoms"])
            if "bonds" in frame:
                self._write_molecule_bonds(f, frame["bonds"])
            if "angles" in frame:
                self._write_molecule_angles(f, frame["angles"])
            if "dihedrals" in frame:
                self._write_molecule_dihedrals(f, frame["dihedrals"])
            if "impropers" in frame:
                self._write_molecule_impropers(f, frame["impropers"])

    def _write_molecule_atoms(self, f, atoms):
        """Write atoms section for molecule format."""
        atoms_df = atoms.to_pandas() if hasattr(atoms, 'to_pandas') else atoms
        
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
        charge_col = None
        if "q" in atoms_df.columns:
            charge_col = "q"
        elif "charge" in atoms_df.columns:
            charge_col = "charge"
            
        if charge_col:
            f.write(f"\nCharges\n\n")
            for i, row in atoms_df.iterrows():
                f.write(f"{i} {float(row[charge_col]):.6f}\n")
        
        # Write molecule IDs
        if "molid" in atoms_df.columns:
            f.write(f"\nMolecules\n\n")
            for i, row in atoms_df.iterrows():
                f.write(f"{i} {int(row['molid'])}\n")

    def _write_molecule_bonds(self, f, bonds):
        """Write bonds section for molecule format."""
        bonds_df = bonds.to_pandas() if hasattr(bonds, 'to_pandas') else bonds
        f.write(f"\nBonds\n\n")
        for i, row in bonds_df.iterrows():
            bond_type = row.get('type', 1)
            atom_i = int(row['i'])
            atom_j = int(row['j'])
            f.write(f"{i} {bond_type} {atom_i} {atom_j}\n")

    def _write_molecule_angles(self, f, angles):
        """Write angles section for molecule format."""
        angles_df = angles.to_pandas() if hasattr(angles, 'to_pandas') else angles
        f.write(f"\nAngles\n\n")
        for i, row in angles_df.iterrows():
            angle_type = row.get('type', 1)
            atom_i = int(row['i'])
            atom_j = int(row['j'])
            atom_k = int(row['k'])
            f.write(f"{i} {angle_type} {atom_i} {atom_j} {atom_k}\n")

    def _write_molecule_dihedrals(self, f, dihedrals):
        """Write dihedrals section for molecule format."""
        dihedrals_df = dihedrals.to_pandas() if hasattr(dihedrals, 'to_pandas') else dihedrals
        f.write(f"\nDihedrals\n\n")
        for i, row in dihedrals_df.iterrows():
            dihedral_type = row.get('type', 1)
            atom_i = int(row['i'])
            atom_j = int(row['j'])
            atom_k = int(row['k'])
            atom_l = int(row['l'])
            f.write(f"{i} {dihedral_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")

    def _write_molecule_impropers(self, f, impropers):
        """Write impropers section for molecule format."""
        impropers_df = impropers.to_pandas() if hasattr(impropers, 'to_pandas') else impropers
        f.write(f"\nImpropers\n\n")
        for i, row in impropers_df.iterrows():
            improper_type = row.get('type', 1)
            atom_i = int(row['i'])
            atom_j = int(row['j'])
            atom_k = int(row['k'])
            atom_l = int(row['l'])
            f.write(f"{i} {improper_type} {atom_i} {atom_j} {atom_k} {atom_l}\n")
