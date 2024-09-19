from pathlib import Path

import numpy as np

import molpy as mp
import io
from itertools import islice
from pyarrow import csv
import pyarrow as pa
import re

class LammpsDataReader:

    def __init__(self, file: str | Path):
        self._file = Path(file)
        self.style = "full"

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r'\s+', ' ', line.strip())

    def read(self):
        
        ff = mp.io.read_forcefield(self._file, format="lammps")

        with open(self._file, "r") as f:

            frame = mp.Frame()

            lines = filter(lambda line: line, map(LammpsDataReader.sanitizer, f))

            box = {
                'xlo': 0.0,
                'xhi': 0.0,
                'ylo': 0.0,
                'yhi': 0.0,
                'zlo': 0.0,
                'zhi': 0.0,
            }

            for line in lines:

                if line.endswith("atoms"):
                    frame["props"]["n_atoms"] = int(line.split()[0])

                elif line.endswith("bonds"):
                    frame["props"]["n_bonds"] = int(line.split()[0])

                elif line.endswith("angles"):
                    frame["props"]["n_angles"] = int(line.split()[0])

                elif line.endswith("dihedrals"):
                    frame["props"]["n_dihedrals"] = int(line.split()[0])

                elif line.endswith("impropers"):
                    frame["props"]["n_impropers"] = int(line.split()[0])

                elif line.endswith("atom types"):
                    frame["props"]["n_atomtypes"] = int(line.split()[0])
                    atom_style = ff.def_atomstyle(self.style)

                elif line.endswith("bond types"):
                    frame["props"]["n_bondtypes"] = int(line.split()[0])
                    bond_style = ff.def_bondstyle("unknown")

                elif line.endswith("angle types"):
                    frame["props"]["n_angletypes"] = int(line.split()[0])
                    angle_style = ff.def_anglestyle("unknown")

                elif line.endswith("dihedral types"):
                    frame["props"]["n_dihedraltypes"] = int(line.split()[0])
                    dihedral_style = ff.def_dihedralstyle("unknown")

                elif line.endswith("improper types"):
                    frame["props"]["n_impropertypes"] = int(line.split()[0])
                    improper_style = ff.def_improperstyle("unknown")

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
                    for line in range(frame["props"]["n_atomtypes"]):
                        line = next(lines).split()
                        atom_type = line[0]
                        if atom_type not in atom_style.types:
                            atom_style.def_type(atom_type, int(atom_type))
                        atom_style.get_type(atom_type).named_params["mass"] = (
                            float(line[1])
                        )

                elif line.startswith("Atoms"):

                    atom_lines = list(islice(lines, frame["props"]["n_atoms"]))
                    probe_line_len = len(atom_lines[0].split())

                    match self.style:

                        case "full":
                            header = ["id", "molid", "type", "charge", "x", "y", "z"]
                            if probe_line_len > 7:
                                if probe_line_len >= 10:
                                    header.append("ix")
                                    header.append("iy")
                                    header.append("iz")
                                if "#" in atom_lines[0].split():
                                    content, sep, comment = atom_lines[0].partition("#")
                                    if comment:
                                        header.append("seq")
                                        header.append("name")
                    atom_table = csv.read_csv(
                        io.BytesIO("\n".join(atom_lines).encode()),
                        read_options=csv.ReadOptions(column_names=header),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                    frame["atoms"] = atom_table

                elif line.startswith("Bonds"):
                    bond_lines = list(islice(lines, frame["props"]["n_bonds"]))
                    bond_table = csv.read_csv(
                        io.BytesIO("\n".join(bond_lines).encode()),
                        read_options=csv.ReadOptions(
                            column_names=["id", "type", "i", "j"]
                        ),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                    frame["bonds"] = bond_table

                elif line.startswith("Angles"):
                    angle_lines = list(islice(lines, frame["props"]["n_angles"]))
                    angle_table = csv.read_csv(
                        io.BytesIO("\n".join(angle_lines).encode()),
                        read_options=csv.ReadOptions(
                            column_names=["id", "type", "i", "j", "k"]
                        ),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                    frame["angles"] = angle_table

                elif line.startswith("Dihedrals"):
                    dihedral_lines = list(islice(lines, frame["props"]["n_dihedrals"]))
                    dihedral_table = csv.read_csv(
                        io.BytesIO("\n".join(dihedral_lines).encode()),
                        read_options=csv.ReadOptions(
                            column_names=["id", "type", "i", "j", "k", "l"]
                        ),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                    frame["dihedrals"] = dihedral_table

                elif line.startswith("Impropers"):
                    improper_lines = list(islice(lines, frame["props"]["n_impropers"]))
                    improper_table = csv.read_csv(
                        io.BytesIO("\n".join(improper_lines).encode()),
                        read_options=csv.ReadOptions(
                            column_names=["id", "type", "i", "j", "k", "l"]
                        ),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                    frame["impropers"] = improper_table

                elif line.startswith("Atom Type Labels"):

                    for line in range(frame["props"]["n_atomtypes"]):
                        line = next(lines).split()
                        atom_type = line[0]
                        if atom_type not in atom_style:
                            at = atom_style.def_type(line[1], int(atom_type))
                            at['id'] = int(atom_type)
                        else:
                            atom_style.get_type(atom_type).name = line[1]

                elif line.startswith("Bond Type Labels"):

                    for line in range(frame["props"]["n_bondtypes"]):
                        line = next(lines).split()
                        bond_type = line[0]
                        if bond_type not in bond_style:
                            bt = bond_style.def_type(line[1])
                            bt['id'] = int(bond_type)
                        else:
                            bond_style.get_type(bond_type).name = line[1]

                elif line.startswith("Angle Type Labels"):
                    for line in range(frame["props"]["n_angletypes"]):
                        line = next(lines).split()
                        angle_type = line[0]
                        if angle_type not in angle_style:
                            at = angle_style.def_type(angle_type)
                            at['id'] = int(angle_type)
                        else:
                            angle_style.get_type(angle_type).name = line[1]

                elif line.startswith("Dihedral Type Labels"):
                    for line in range(frame["props"]["n_dihedraltypes"]):
                        line = next(lines).split()
                        dihedral_type = line[0]
                        if dihedral_type not in dihedral_style:
                            dt = dihedral_style.def_type(line[1])
                            dt['id'] = int(dihedral_type)
                        else:
                            dihedral_style.get_type(dihedral_type).name = line[1]

                elif line.startswith("Improper Type Labels"):
                    for line in range(frame["props"]["n_impropertypes"]):
                        line = next(lines).split()
                        improper_type = line[0]
                        if improper_type not in improper_style:
                            it = improper_style.def_type(line[1])
                            it['id'] = int(improper_type)
                        else:
                            improper_style.get_type(improper_type).name = line[1]

        return mp.System(
            mp.Box(
                np.diag(
                    [
                        box["xhi"] - box["xlo"],
                        box["yhi"] - box["ylo"],
                        box["zhi"] - box["zlo"],
                    ]
                )
            ),
            ff,
            frame,
        )


class LammpsDataWriter:

    def __init__(self, frame: mp.Frame, file: str | Path, atom_style: str = "full"):
        self._frame = frame
        self._file = Path(file)
        self._atom_style = atom_style

    def write(self):

        frame = self._frame

        with open(self._file, "w") as f:

            if "xyz" in frame["atoms"]:
                xyz = frame["atoms"]["xyz"]
            if "charge" in frame["atoms"]:
                q = frame["atoms"]["charge"]
            if "molid" in frame["atoms"]:
                molid = frame["atoms"]["molid"]
            if "type" in frame["atoms"]:
                type = frame["atoms"]["type"]
            if "id" in frame["atoms"]:
                id = frame["atoms"]["id"]

            f.write(f"# generated by molpy\n\n")
            f.write(f"{frame['props']['n_atoms']} atoms\n")
            f.write(f"{frame['props']['n_bonds']} bonds\n")
            f.write(f"{frame['props']['n_angles']} angles\n")
            f.write(f"{frame['props']['n_dihedrals']} dihedrals\n\n")
            f.write(f"{frame['props']['n_atomtypes']} atom types\n")
            f.write(f"{frame['props']['n_bondtypes']} bond types\n")
            f.write(f"{frame['props']['n_angletypes']} angle types\n")
            f.write(f"{frame['props']['n_dihedraltypes']} dihedral types\n\n")

            f.write(f"{frame.box.xlo} {frame.box.xhi} xlo xhi\n")
            f.write(f"{frame.box.ylo} {frame.box.yhi} ylo yhi\n")
            f.write(f"{frame.box.zlo} {frame.box.zhi} zlo zhi\n\n")

            if "mass" in frame["atoms"]:

                f.write(f"Masses\n\n")
                unique_types, unique_indices = np.unique(type, return_index=True)
                unique_masses = frame["atoms"]["mass"][unique_indices]
                for i, mass in zip(unique_types, unique_masses):
                    f.write(f"{i} {mass}\n")

            f.write(f"Atoms\n\n")
            match self._atom_style:
                case "full":
                    for id, molid, type, q, r in zip(id, molid, type, q, xyz):
                        f.write(f"{id} {molid} {type} {q} {r[0]} {r[1]} {r[2]}\n")

            f.write(f"\nBonds\n\n")
            for id, i, j, type in zip(
                frame["bonds"]["id"],
                frame["bonds"]["i"],
                frame["bonds"]["j"],
                frame["bonds"]["type"],
            ):
                f.write(f"{id} {type} {i} {j}\n")

            f.write(f"\nAngles\n\n")
            for id, type, i, j, k in zip(
                frame["angles"]["id"],
                frame["angles"]["type"],
                frame["angles"]["i"],
                frame["angles"]["j"],
                frame["angles"]["k"],
            ):
                f.write(f"{id} {type} {i} {j} {k}\n")

            f.write(f"\nDihedrals\n\n")
            for id, type, i, j, k, l in zip(
                frame["dihedrals"]["id"],
                frame["dihedrals"]["type"],
                frame["dihedrals"]["i"],
                frame["dihedrals"]["j"],
                frame["dihedrals"]["k"],
                frame["dihedrals"]["l"],
            ):
                f.write(f"{id} {type} {i} {j} {k} {l}\n")
