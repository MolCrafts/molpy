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

    def read(self, system: mp.System):
        
        ff = system.forcefield
        if ff.n_atomstyles == 0:
            atom_style = ff.def_atomstyle("full")
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
                    # bond_style = ff.def_bondstyle("unknown")

                elif line.endswith("angle types"):
                    frame["props"]["n_angletypes"] = int(line.split()[0])
                    # angle_style = ff.def_anglestyle("unknown")

                elif line.endswith("dihedral types"):
                    frame["props"]["n_dihedraltypes"] = int(line.split()[0])
                    # dihedral_style = ff.def_dihedralstyle("unknown")

                elif line.endswith("improper types"):
                    frame["props"]["n_impropertypes"] = int(line.split()[0])
                    # improper_style = ff.def_improperstyle("unknown")

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
                    masses = {}
                    for line in range(frame["props"]["n_atomtypes"]):
                        line = next(lines).split()
                        masses[int(line[0])] = float(line[1])

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
                        atom_type = int(line[0])
                        if atom_type not in atom_style:
                            at = atom_style.def_type(line[1], kw_params={'id': atom_type})
                        else:
                            atom_style[atom_type].name = line[1]

                elif line.startswith("Bond Type Labels"):
                    if not bond_style:
                        continue
                    for line in range(frame["props"]["n_bondtypes"]):
                        line = next(lines).split()
                        bond_type = int(line[0])
                        if bond_type not in bond_style:
                            bt = bond_style.def_type(line[1], kw_params={'id': bond_type})
                        else:
                            bond_style[bond_type].name = line[1]

                elif line.startswith("Angle Type Labels"):
                    if not angle_style:
                        continue
                    for line in range(frame["props"]["n_angletypes"]):
                        line = next(lines).split()
                        angle_type = int(line[0])
                        if angle_type not in angle_style:
                            at = angle_style.def_type(line[1], kw_params={'id': angle_type})
                        else:
                            angle_style[angle_type].name = line[1]

                elif line.startswith("Dihedral Type Labels"):
                    if not dihedral_style:
                        continue
                    for line in range(frame["props"]["n_dihedraltypes"]):
                        line = next(lines).split()
                        dihedral_type = int(line[0])
                        if dihedral_type not in dihedral_style:
                            dt = dihedral_style.def_type(line[1], kw_params={'id': dihedral_type})
                        else:
                            dihedral_style[dihedral_type].name = line[1]

                elif line.startswith("Improper Type Labels"):
                    if not improper_style:
                        continue
                    for line in range(frame["props"]["n_impropertypes"]):
                        line = next(lines).split()
                        improper_type = int(line[0])
                        if improper_type not in improper_style:
                            it = improper_style.def_type(line[1], kw_params={'id': improper_type})
                        else:
                            improper_style[improper_type].name = line[1]

        per_atom_mass = np.zeros(frame["props"]["n_atoms"])
        for t, m in masses.items():
            atomtype = atom_style.get_by(lambda atom: atom['id'] == t)
            if not atomtype:
                atomtype = atom_style.def_type(str(t), kw_params={'id': t})
            atomtype['mass'] = m
            
            per_atom_mass[frame["atoms"]["type"] == t] = m
        frame["atoms"] = frame["atoms"].append_column("mass", pa.array(per_atom_mass))

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

    def __init__(self, file: str | Path, atom_style: str = "full"):
        self._file = Path(file)
        self._atom_style = atom_style

    def write(self, system: mp.System):

        frame = system.frame
        ff = system.forcefield

        with open(self._file, "w") as f:

            f.write(f"# generated by molpy\n\n")
            f.write(f"{len(frame['atoms'])} atoms\n")
            f.write(f"{len(frame['bonds'])} bonds\n")
            f.write(f"{len(frame['angles'])} angles\n")
            f.write(f"{len(frame['dihedrals'])} dihedrals\n\n")
            f.write(f"{len(np.unique(frame['atoms']['type']))} atom types\n")
            f.write(f"{len(np.unique(frame['bonds']['type']))} bond types\n")
            f.write(f"{len(np.unique(frame['angles']['type']))} angle types\n")
            f.write(f"{len(np.unique(frame['dihedrals']['type']))} dihedral types\n\n")

            xlo = system.box.xlo
            xhi = system.box.xhi
            ylo = system.box.ylo
            yhi = system.box.yhi
            zlo = system.box.zlo
            zhi = system.box.zhi

            f.write(f"{xlo:.2f} {xhi:.2f} xlo xhi\n")
            f.write(f"{ylo:.2f} {yhi:.2f} ylo yhi\n")
            f.write(f"{zlo:.2f} {zhi:.2f} zlo zhi\n\n")

            if "type_name" in frame["atoms"].column_names:
                f.write(f"\nAtom Type Labels\n\n")
                atomtypenames = frame["atoms"]["type_name"].to_numpy()
                unique_types = np.unique(atomtypenames)
                for i, at in enumerate(unique_types, 1):
                    f.write(f"{i} {at}\n")

            if "type_name" in frame["bonds"].column_names:
                f.write(f"\nBond Type Labels\n\n")
                bondtypenames = frame["bonds"]["type_name"].to_numpy()
                unique_types = np.unique(bondtypenames)
                for i, bt in enumerate(unique_types, 1):
                    f.write(f"{i} {bt}\n")

            if "type_name" in frame["angles"].column_names:
                f.write(f"\nAngle Type Labels\n\n")
                angletypenames = frame["angles"]["type_name"].to_numpy()
                unique_types = np.unique(angletypenames)
                for i, at in enumerate(unique_types, 1):
                    f.write(f"{i} {at}\n")

            if "type_name" in frame["dihedrals"].column_names:
                f.write(f"\nDihedral Type Labels\n\n")
                dihedraltypenames = frame["dihedrals"]["type_name"].to_numpy()
                unique_types = np.unique(dihedraltypenames)
                for i, dt in enumerate(unique_types, 1):
                    f.write(f"{i} {dt}\n")

            if "mass" in frame["atoms"].column_names:

                f.write(f"\nMasses\n\n")
                for atomtype in ff.atomtypes.values():
                    f.write(f"{atomtype.name} {atomtype['mass']:.2f}\n")

            f.write(f"\nAtoms\n\n")

            match self._atom_style:
                case "full":
                    for id, molid, type, q, x, y, z in zip(
                        frame["atoms"]["id"],
                        frame["atoms"]["molid"],
                        frame["atoms"]['type'],
                        frame["atoms"]["charge"],
                        frame["atoms"]["x"],
                        frame["atoms"]["y"],
                        frame["atoms"]["z"],
                    ):
                        f.write(f"{id} {molid} {type} {q.as_py():.2f} {x.as_py():.2f} {y.as_py():.2f} {z.as_py():.2f}\n")

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

class LammpsMoleculeReader:

    def __init__(self, file: str | Path):
        self._file = Path(file)
        self.style = "full"
        with open(self._file, "r") as f:
            self.lines = filter(lambda line: line, map(LammpsDataReader.sanitizer, f))

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r'\s+', ' ', line.strip())

    def read(self, system: mp.System):

        frame = system.frame

        for line in self.lines:

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

            elif line.startswith('Corrds'):
                header = ["id", 'x', 'y', 'z']
                atom_lines = list(islice(self.lines, frame["props"]["n_atoms"]))
                atom_table = csv.read_csv(
                        io.BytesIO("\n".join(atom_lines).encode()),
                        read_options=csv.ReadOptions(column_names=header),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                frame["atoms"] = atom_table

            elif line.startswith('Types'):
                header = ["id", 'type']
                atomtype_lines = list(islice(self.lines, frame["props"]["n_atoms"]))
                atomtype_table = csv.read_csv(
                        io.BytesIO("\n".join(atomtype_lines).encode()),
                        read_options=csv.ReadOptions(column_names=header),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                # join atom table and type table
                frame["atoms"] = frame["atoms"].join(atomtype_table, on='id')
                self._read_line(line, frame)

            elif line.startswith("Charges"):
                header = ["id", 'charge']
                charge_lines = list(islice(self.lines, frame["props"]["n_atoms"]))
                charge_table = csv.read_csv(
                        io.BytesIO("\n".join(charge_lines).encode()),
                        read_options=csv.ReadOptions(column_names=header),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                frame["atoms"] = frame["atoms"].join(charge_table, on='id')
                
            elif line.startswith("Molecules"):
                header = ["id", 'molid']
                molid_lines = list(islice(self.lines, frame["props"]["n_atoms"]))
                molid_table = csv.read_csv(
                        io.BytesIO("\n".join(molid_lines).encode()),
                        read_options=csv.ReadOptions(column_names=header),
                        parse_options=csv.ParseOptions(delimiter=" "),
                    )
                frame["atoms"] = frame["atoms"].join(molid_table, on='id')

            elif line.startswith("Bonds"):
                bond_lines = list(islice(self.lines, frame["props"]["n_bonds"]))
                bond_table = csv.read_csv(
                    io.BytesIO("\n".join(bond_lines).encode()),
                    read_options=csv.ReadOptions(
                        column_names=["id", "type", "i", "j"]
                    ),
                    parse_options=csv.ParseOptions(delimiter=" "),
                )
                frame["bonds"] = bond_table

            elif line.startswith("Angles"):
                angle_lines = list(islice(self.lines, frame["props"]["n_angles"]))
                angle_table = csv.read_csv(
                    io.BytesIO("\n".join(angle_lines).encode()),
                    read_options=csv.ReadOptions(
                        column_names=["id", "type", "i", "j", "k"]
                    ),
                    parse_options=csv.ParseOptions(delimiter=" "),
                )
                frame["angles"] = angle_table

            elif line.startswith("Dihedrals"):
                dihedral_lines = list(islice(self.lines, frame["props"]["n_dihedrals"]))
                dihedral_table = csv.read_csv(
                    io.BytesIO("\n".join(dihedral_lines).encode()),
                    read_options=csv.ReadOptions(
                        column_names=["id", "type", "i", "j", "k", "l"]
                    ),
                    parse_options=csv.ParseOptions(delimiter=" "),
                )
                frame["dihedrals"] = dihedral_table

            elif line.startswith("Impropers"):
                improper_lines = list(islice(self.lines, frame["props"]["n_impropers"]))
                improper_table = csv.read_csv(
                    io.BytesIO("\n".join(improper_lines).encode()),
                    read_options=csv.ReadOptions(
                        column_names=["id", "type", "i", "j", "k", "l"]
                    ),
                    parse_options=csv.ParseOptions(delimiter=" "),
                )
                frame["impropers"] = improper_table

class LammpsMoleculeWriter:

    def __init__(self, file: str | Path, atom_style: str = "full"):
        self._file = Path(file)
        self._atom_style = atom_style

    def write(self, system):

        frame = system.frame

        with open(self._file, "w") as f:

            f.write(f"# generated by molpy\n\n")

            if 'n_atoms' in frame['props']:
                f.write(f"{frame['props']['n_atoms']} atoms\n")
            if 'n_bonds' in frame['props']:
                f.write(f"{frame['props']['n_bonds']} bonds\n")
            if 'n_angles' in frame['props']:
                f.write(f"{frame['props']['n_angles']} angles\n")
            if 'n_dihedrals' in frame['props']:
                f.write(f"{frame['props']['n_dihedrals']} dihedrals\n")

            if 'atoms' in frame:
                atoms = frame['atoms'].to_pylist()
                f.write(f"\nCoords\n\n")
                for atom in atoms:
                    f.write(f"{atom['id']} {atom['x']} {atom['y']} {atom['z']}\n")

                f.write(f"\nTypes\n\n")
                for atom in atoms:
                    f.write(f"{atom['id']} {atom['type_name'] or atom['type']}\n")

                if 'charge' in frame["atoms"]:
                    f.write(f"\nCharges\n\n")
                    for atom in atoms:
                        f.write(f"{atom['id']} {atom['charge']}\n")

                if 'molid' in frame["atoms"]:
                    f.write(f"\nMolecules\n\n")
                    for atom in atoms:
                        f.write(f"{atom['id']} {atom['molid']}\n")

            if 'bonds' in frame:
                bonds = frame['bonds'].to_pylist()
                f.write(f"\nBonds\n\n")
                for bond in bonds:
                    f.write(f"{bond['id']} {bond['type_name']} {bond['i']} {bond['j']}\n")

            if 'angles' in frame:
                angles = frame['angles'].to_pylist()
                f.write(f"\nAngles\n\n")
                for angle in angles:
                    f.write(f"{angle['id']} {angle['type_name']} {angle['i']} {angle['j']} {angle['k']}\n")

            if 'dihedrals' in frame:
                dihedrals = frame['dihedrals'].to_pylist()
                f.write(f"\nDihedrals\n\n")
                for dihedral in dihedrals:
                    f.write(f"{dihedral['id']} {dihedral['type_name']} {dihedral['i']} {dihedral['j']} {dihedral['k']} {dihedral['l']}\n")

            if 'impropers' in frame:
                impropers = frame['impropers'].to_pylist()
                f.write(f"\nImpropers\n\n")
                for improper in impropers:
                    f.write(f"{improper['id']} {improper['type_name']} {improper['i']} {improper['j']} {improper['k']} {improper['l']}\n")

            if "molid" in frame["atoms"]:
                f.write(f"\nMolecule\n\n")
                for i, molid in zip(frame["atoms"]["id"], frame["atoms"]["molid"]):
                    f.write(f"{i} {molid}\n")
