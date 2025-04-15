from pathlib import Path

import numpy as np

import molpy as mp
import io
from itertools import islice
import pandas as pd
import re
from .base import DataReader, DataWriter


class LammpsDataReader(DataReader):

    def __init__(self, path: str | Path, atom_style="full"):
        super().__init__(path)
        self.atom_style = atom_style
        self._file = open(path, "r")

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

        lines = filter(lambda line: line, map(LammpsDataReader.sanitizer, self._file))

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

        for line in lines:

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
                for line in range(props["n_atomtypes"]):
                    line = next(lines).split()
                    masses[line[0]] = float(line[1])

            elif line.startswith("Atoms"):

                atom_lines = list(islice(lines, props["n_atoms"]))
                probe_line_len = len(atom_lines[0].split())

                match self.atom_style:

                    case "full":
                        header = ["id", "molid", type_key, "charge", "x", "y", "z"]
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
                atom_table = pd.read_csv(
                    io.BytesIO("\n".join(atom_lines).encode()),
                    names=header,
                    delimiter=" ",
                )
                if masses:
                    atom_table["mass"] = atom_table[type_key].map(
                        lambda t: masses[str(t)]
                    )

                frame["atoms"] = atom_table

            elif line.startswith("Bonds"):
                bond_lines = list(islice(lines, props["n_bonds"]))
                bond_table = pd.read_csv(
                    io.BytesIO("\n".join(bond_lines).encode()),
                    names=["id", type_key, "i", "j"],
                    delimiter=" ",
                )
                frame["bonds"] = bond_table

            elif line.startswith("Angles"):
                angle_lines = list(islice(lines, props["n_angles"]))
                angle_table = pd.read_csv(
                    io.BytesIO("\n".join(angle_lines).encode()),
                    names=["id", type_key, "i", "j", "k"],
                    delimiter=" ",
                )
                frame["angles"] = angle_table

            elif line.startswith("Dihedrals"):
                dihedral_lines = list(islice(lines, props["n_dihedrals"]))
                dihedral_table = pd.read_csv(
                    io.BytesIO("\n".join(dihedral_lines).encode()),
                    names=["id", type_key, "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["dihedrals"] = dihedral_table

            elif line.startswith("Impropers"):
                improper_lines = list(islice(lines, props["n_impropers"]))
                improper_table = pd.read_csv(
                    io.BytesIO("\n".join(improper_lines).encode()),
                    names=["id", type_key, "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["impropers"] = improper_table

            elif line.startswith("Atom Type Labels"):
                type_key = "type"
                for line in range(props["n_atomtypes"]):
                    line = next(lines).split()
                    atom_type = int(line[0])
                    if atom_type not in atom_style:
                        at = atom_style.def_type(
                            line[1], kw_params={"id": atom_type}
                        )
                    else:
                        atom_style[atom_type].name = line[1]

            elif line.startswith("Bond Type Labels"):
                if not bond_style:
                    continue
                for line in range(props["n_bondtypes"]):
                    line = next(lines).split()
                    bond_type = int(line[0])
                    if bond_type not in bond_style:
                        bt = bond_style.def_type(
                            line[1], kw_params={"id": bond_type}
                        )
                    else:
                        bond_style[bond_type].name = line[1]

            elif line.startswith("Angle Type Labels"):
                if not angle_style:
                    continue
                for line in range(props["n_angletypes"]):
                    line = next(lines).split()
                    angle_type = int(line[0])
                    if angle_type not in angle_style:
                        at = angle_style.def_type(
                            line[1], kw_params={"id": angle_type}
                        )
                    else:
                        angle_style[angle_type].name = line[1]

            elif line.startswith("Dihedral Type Labels"):
                if not dihedral_style:
                    continue
                for line in range(props["n_dihedraltypes"]):
                    line = next(lines).split()
                    dihedral_type = int(line[0])
                    if dihedral_type not in dihedral_style:
                        dt = dihedral_style.def_type(
                            line[1], kw_params={"id": dihedral_type}
                        )
                    else:
                        dihedral_style[dihedral_type].name = line[1]

            elif line.startswith("Improper Type Labels"):
                if not improper_style:
                    continue
                for line in range(props["n_impropertypes"]):
                    line = next(lines).split()
                    improper_type = int(line[0])
                    if improper_type not in improper_style:
                        it = improper_style.def_type(
                            line[1], kw_params={"id": improper_type}
                        )
                    else:
                        improper_style[improper_type].name = line[1]

        per_atom_mass = np.zeros(props["n_atoms"])
        for t, m in masses.items():
            atomtype = atom_style.get_by(lambda atom: atom.label == str(t))
            if not atomtype:
                atomtype = atom_style.def_type(str(t), kw_params={"id": t})
            atomtype["mass"] = m

            per_atom_mass[frame["atoms"][type_key] == t] = m  # todo: type mapping
        frame["atoms"]["mass"] = per_atom_mass

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

    def __init__(self, path: str | Path, atom_style: str = "full", reindex: bool = True, mode="w"):
        super().__init__(path, mode=mode)
        self._atom_style = atom_style
        self.reindex = reindex

    def write(self, frame):

        system = self._system
        frame = system.frame
        ff = system.forcefield


        n_atoms = len(frame["atoms"])
        n_bonds = len(frame["bonds"]) if "bonds" in frame else 0
        n_angles = len(frame["angles"]) if "angles" in frame else 0
        n_dihedrals = len(frame["dihedrals"]) if "dihedrals" in frame else 0

        f = self._file

        f.write(f"# generated by molpy\n\n")
        f.write(f"{n_atoms} atoms\n")
        f.write(f"{system.n_atomtypes} atom types\n")
        if "bonds" in frame:
            f.write(f"{n_bonds} bonds\n")
            f.write(f"{system.n_bondtypes} bond types\n")
        if "angles" in frame:
            f.write(f"{n_angles} angles\n")
            f.write(f"{system.n_angletypes} angle types\n")
        if "dihedrals" in frame:
            f.write(f"{n_dihedrals} dihedrals\n")
            f.write(f"{system.n_dihedraltypes} dihedral types\n\n")

        xlo = frame.box.xlo
        xhi = frame.box.xhi
        ylo = frame.box.ylo
        yhi = frame.box.yhi
        zlo = frame.box.zlo
        zhi = frame.box.zhi

        f.write(f"{xlo:.2f} {xhi:.2f} xlo xhi\n")
        f.write(f"{ylo:.2f} {yhi:.2f} ylo yhi\n")
        f.write(f"{zlo:.2f} {zhi:.2f} zlo zhi\n")

        type_key = "type"

        if ff.atomstyles:

            found_type_label_in_atom = "type" in frame["atoms"]
            type_key = "type" if found_type_label_in_atom else "type"

            if type_key == "type":
                f.write(f"\nAtom Type Labels\n\n")
                for i, atomtype in enumerate(ff.get_atomtypes(), 1):
                    f.write(f"{i} {atomtype.name}\n")

            if ff.bondstyles:
            
                f.write(f"\nBond Type Labels\n\n")
                for i, bondtype in enumerate(ff.get_bondtypes(), 1):
                    f.write(f"{i} {bondtype.name}\n")

            if ff.anglestyles:
            
                f.write(f"\nAngle Type Labels\n\n")
                for i, angletype in enumerate(ff.get_angletypes(), 1):
                    f.write(f"{i} {angletype.name}\n")

            if ff.dihedralstyles:
            
                f.write(f"\nDihedral Type Labels\n\n")
                for i, dihedraltype in enumerate(ff.get_dihedraltypes(), 1):
                    f.write(f"{i} {dihedraltype.name}\n")

        f.write(f"\nMasses\n\n")
        masses = {}
        try:
            if ff.n_atomtypes:
                for atomtype in ff.get_atomtypes():
                    masses[atomtype.name] = atomtype["mass"]
            else:
                raise KeyError("n_atomtypes")
        except KeyError:
            masses = {}
            unique_type, unique_idx = np.unique(
                frame["atoms"]["type"].to_numpy(), return_index=True
            )
            mass_arr = frame["atoms"]["mass"].to_numpy()
            for i, t in zip(unique_idx, unique_type):
                # f.write(f"{frame['atoms']['type'][i]} {m:.3f}\n")
                masses[t] = mass_arr[i]

        for i, m in masses.items():
            f.write(f"{i} {m:.3f}\n")

        if self.reindex:
            # 旧的 atom id 映射
            new_id = np.arange(1, n_atoms + 1)
            _old_atom_idx = pd.Series(new_id, index=frame["atoms"]["id"])

            # 重新编号 atom id
            frame["atoms"]["id"] = new_id

            # 处理 bonds
            if "bonds" in frame:
                frame["bonds"]["i"] = frame["bonds"]["i"].map(_old_atom_idx)
                frame["bonds"]["j"] = frame["bonds"]["j"].map(_old_atom_idx)
                frame["bonds"]["id"] = np.arange(1, n_bonds + 1)

            # 处理 angles
            if "angles" in frame:
                frame["angles"]["i"] = frame["angles"]["i"].map(_old_atom_idx)
                frame["angles"]["j"] = frame["angles"]["j"].map(_old_atom_idx)
                frame["angles"]["k"] = frame["angles"]["k"].map(_old_atom_idx)
                frame["angles"]["id"] = np.arange(1, n_angles + 1)

            # 处理 dihedrals
            if "dihedrals" in frame:
                frame["dihedrals"]["i"] = frame["dihedrals"]["i"].map(_old_atom_idx)
                frame["dihedrals"]["j"] = frame["dihedrals"]["j"].map(_old_atom_idx)
                frame["dihedrals"]["k"] = frame["dihedrals"]["k"].map(_old_atom_idx)
                frame["dihedrals"]["l"] = frame["dihedrals"]["l"].map(_old_atom_idx)
                frame["dihedrals"]["id"] = np.arange(1, n_dihedrals + 1)


        f.write(f"\nAtoms\n\n")

        match self._atom_style:
            case "full":
                for id, molid, type, q, x, y, z in zip(
                    frame["atoms"]["id"],
                    frame["atoms"]["molid"],
                    frame["atoms"][type_key],
                    frame["atoms"]["charge"],
                    frame["atoms"]["x"],
                    frame["atoms"]["y"],
                    frame["atoms"]["z"],
                ):
                    f.write(
                        f"{id} {int(molid)} {type} {q:.3f} {x:.3f} {y:.3f} {z:.3f}\n"
                    )

        f.write(f"\nBonds\n\n")
        for id, i, j, type in zip(
            frame["bonds"]["id"],
            frame["bonds"]["i"],
            frame["bonds"]["j"],
            frame["bonds"][type_key],
        ):
            f.write(f"{id} {type} {i} {j}\n")

        f.write(f"\nAngles\n\n")
        for id, type, i, j, k in zip(
            frame["angles"]["id"],
            frame["angles"][type_key],
            frame["angles"]["i"],
            frame["angles"]["j"],
            frame["angles"]["k"],
        ):
            f.write(f"{id} {type} {i} {j} {k}\n")

        f.write(f"\nDihedrals\n\n")
        for id, type, i, j, k, l in zip(
            frame["dihedrals"]["id"],
            frame["dihedrals"][type_key],
            frame["dihedrals"]["i"],
            frame["dihedrals"]["j"],
            frame["dihedrals"]["k"],
            frame["dihedrals"]["l"],
        ):
            f.write(f"{id} {type} {i} {j} {k} {l}\n")


class LammpsMoleculeReader(DataReader):

    def __init__(self, file: str | Path, system: mp.System | None = None):
        super().__init__(file, system)
        self._path = Path(file)
        self.style = "full"
        with open(self._path, "r") as f:
            self.lines = filter(lambda line: line, map(LammpsDataReader.sanitizer, f))

    @staticmethod
    def sanitizer(line: str) -> str:
        return re.sub(r"\s+", " ", line.strip())

    def read(self, system: mp.System):

        frame = system.frame

        props = {}

        for line in self.lines:

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

            elif line.startswith("Corrds"):
                header = ["id", "x", "y", "z"]
                atom_lines = list(islice(self.lines, props["n_atoms"]))
                atom_table = pd.read_csv(
                    io.BytesIO("\n".join(atom_lines).encode()),
                    names=header,
                    delimiter=" ",
                )
                frame["atoms"] = atom_table

            elif line.startswith("Types"):
                header = ["id", "type"]
                atomtype_lines = list(islice(self.lines, props["n_atoms"]))
                atomtype_table = pd.read_csv(
                    io.BytesIO("\n".join(atomtype_lines).encode()),
                    names=header,
                    delimiter=" ",
                )
                # join atom table and type table
                frame["atoms"] = frame["atoms"].join(atomtype_table, on="id")
                self._read_line(line, frame)

            elif line.startswith("Charges"):
                header = ["id", "charge"]
                charge_lines = list(islice(self.lines, props["n_atoms"]))
                charge_table = pd.read_csv(
                    io.BytesIO("\n".join(charge_lines).encode()),
                    names=header,
                    delimiter=" ",
                )
                frame["atoms"] = frame["atoms"].join(charge_table, on="id")

            elif line.startswith("Molecules"):
                header = ["id", "molid"]
                molid_lines = list(islice(self.lines, props["n_atoms"]))
                molid_table = pd.read_csv(
                    io.BytesIO("\n".join(molid_lines).encode()),
                    names=header,
                    delimiter=" ",
                )
                frame["atoms"] = frame["atoms"].join(molid_table, on="id")

            elif line.startswith("Bonds"):
                bond_lines = list(islice(self.lines, props["n_bonds"]))
                bond_table = pd.read_csv(
                    io.BytesIO("\n".join(bond_lines).encode()),
                    names=["id", "type", "i", "j"],
                    delimiter=" ",
                )
                frame["bonds"] = bond_table

            elif line.startswith("Angles"):
                angle_lines = list(islice(self.lines, props["n_angles"]))
                angle_table = pd.read_csv(
                    io.BytesIO("\n".join(angle_lines).encode()),
                    names=["id", "type", "i", "j", "k"],
                    delimiter=" ",
                )
                frame["angles"] = angle_table

            elif line.startswith("Dihedrals"):
                dihedral_lines = list(islice(self.lines, props["n_dihedrals"]))
                dihedral_table = pd.read_csv(
                    io.BytesIO("\n".join(dihedral_lines).encode()),
                    names=["id", "type", "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["dihedrals"] = dihedral_table

            elif line.startswith("Impropers"):
                improper_lines = list(islice(self.lines, props["n_impropers"]))
                improper_table = pd.read_csv(
                    io.BytesIO("\n".join(improper_lines).encode()),
                    names=["id", "type", "i", "j", "k", "l"],
                    delimiter=" ",
                )
                frame["impropers"] = improper_table


class LammpsMoleculeWriter:

    def __init__(self, file: str | Path, atom_style: str = "full"):
        self._path = Path(file)
        self._atom_style = atom_style

    def write(self, system):

        frame = system.frame

        with open(self._path, "w") as f:

            f.write(f"# generated by molpy\n\n")

            n_atoms = len(frame["atoms"])
            f.write(f"{n_atoms} atoms\n")
            n_bonds = len(frame["bonds"]) if "bonds" in frame else 0
            f.write(f"{n_bonds} bonds\n")
            n_angles = len(frame["angles"]) if "angles" in frame else 0
            f.write(f"{n_angles} angles\n")
            n_dihedrals = len(frame["dihedrals"]) if "dihedrals" in frame else 0
            f.write(f"{n_dihedrals} dihedrals\n\n")

            if "atoms" in frame:
                f.write(f"\nCoords\n\n")
                for i, atom in frame["atoms"].iterrows():
                    f.write(f"{i} {atom['x']} {atom['y']} {atom['z']}\n")

                f.write(f"\nTypes\n\n")
                for i, atom in frame["atoms"].iterrows():
                    f.write(f"{i} {atom['type']}\n")

                if "charge" in frame["atoms"]:
                    f.write(f"\nCharges\n\n")
                    for i, atom in frame["atoms"].iterrows():
                        f.write(f"{i} {atom['charge']:.3f}\n")

                if "molid" in frame["atoms"]:
                    f.write(f"\nMolecules\n\n")
                    for i, atom in frame["atoms"].iterrows():
                        f.write(f"{i} {atom['molid']}\n")

            if "bonds" in frame:
                bonds = frame["bonds"].iterrows()
                f.write(f"\nBonds\n\n")
                for i, bond in bonds:
                    f.write(f"{i} {bond['type']} {bond['i']} {bond['j']}\n")

            if "angles" in frame:
                angles = frame["angles"].iterrows()
                f.write(f"\nAngles\n\n")
                for i, angle in angles:
                    f.write(
                        f"{i} {angle['type']} {angle['i']} {angle['j']} {angle['k']}\n"
                    )

            if "dihedrals" in frame:
                dihedrals = frame["dihedrals"].iterrows()
                f.write(f"\nDihedrals\n\n")
                for i, dihedral in dihedrals:
                    f.write(
                        f"{i} {dihedral['type']} {dihedral['i']} {dihedral['j']} {dihedral['k']} {dihedral['l']}\n"
                    )

            if "impropers" in frame:
                impropers = frame["impropers"].iterrows()
                f.write(f"\nImpropers\n\n")
                for i, improper in impropers:
                    f.write(
                        f"{i} {improper['type']} {improper['i']} {improper['j']} {improper['k']} {improper['l']}\n"
                    )
