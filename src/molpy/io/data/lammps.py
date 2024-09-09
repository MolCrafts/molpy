from pathlib import Path

import numpy as np

import molpy as mp


class LammpsDataReader:

    def __init__(self, file: str | Path):
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    def read(self):

        with open(self._file, "r") as f:

            frame = mp.Frame()

            lines = filter(lambda line: line, map(LammpsDataReader.sanitizer, f))

            box = {}

            for line in lines:

                if line.endswith("atoms"):
                    frame["global_props"]["n_atoms"] = int(line.split()[0])

                elif line.endswith("bonds"):
                    frame["global_props"]["n_bonds"] = int(line.split()[0])

                elif line.endswith("angles"):
                    frame["global_props"]["n_angles"] = int(line.split()[0])

                elif line.endswith("dihedrals"):
                    frame["global_props"]["n_dihedrals"] = int(line.split()[0])

                elif line.endswith("atom types"):
                    frame["global_props"]["n_atomtypes"] = int(line.split()[0])

                elif line.endswith("bond types"):
                    frame["global_props"]["n_bondtypes"] = int(line.split()[0])

                elif line.endswith("angle types"):
                    frame["global_props"]["n_angletypes"] = int(line.split()[0])

                elif line.endswith("dihedral types"):
                    frame["global_props"]["n_dihedraltypes"] = int(line.split()[0])

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
                    for line in range(frame["global_props"]["n_atomtypes"]):
                        line = next(lines).split()
                        masses[int(line[0])] = float(line[1])

                elif line.startswith("Atoms"):

                    ATOM, sep, style = line.partition("#")
                    style = style.strip()
                    if style.strip() == "full":
                        atoms = {
                            "id": [],
                            "molid": [],
                            "type": [],
                            "charge": [],
                            "xyz": [],
                        }

                    for i in range(frame["global_props"]["n_atoms"]):
                        match style:
                            case "full":
                                id_, molid, type_, charge, x, y, z = next(lines).split()
                                atoms["id"].append(int(id_))
                                atoms["molid"].append(int(molid))
                                atoms["type"].append(int(type_))
                                atoms["charge"].append(float(charge))
                                atoms["xyz"].append((float(x), float(y), float(z)))

                    for key, value in atoms.items():
                        frame["atoms"][key] = np.array(value)

                elif line.startswith("Bonds"):
                    bonds = []
                    for line in range(frame["global_props"]["n_bonds"]):
                        line = next(lines).split()
                        bonds.append(list(map(int, line)))

                    bonds = np.array(bonds)
                    frame["bonds"]["id"] = bonds[:, 0]
                    frame["bonds"]["type"] = bonds[:, 1]
                    frame["bonds"]["i"] = bonds[:, 2]
                    frame["bonds"]["j"] = bonds[:, 3]

                elif line.startswith("Angles"):
                    angles = []
                    for line in range(frame["global_props"]["n_angles"]):
                        line = next(lines).split()
                        angles.append(list(map(int, line)))

                    angles = np.array(angles)
                    frame["angles"]["id"] = angles[:, 0]
                    frame["angles"]["type"] = angles[:, 1]
                    frame["angles"]["i"] = angles[:, 2]
                    frame["angles"]["j"] = angles[:, 3]
                    frame["angles"]["k"] = angles[:, 4]

                elif line.startswith("Dihedrals"):
                    dihedrals = []
                    for line in range(frame["global_props"]["n_dihedrals"]):
                        line = next(lines).split()
                        dihedrals.append(list(map(int, line)))

                    dihedrals = np.array(dihedrals)
                    frame["dihedrals"]["id"] = dihedrals[:, 0]
                    frame["dihedrals"]["type"] = dihedrals[:, 1]
                    frame["dihedrals"]["i"] = dihedrals[:, 2]
                    frame["dihedrals"]["j"] = dihedrals[:, 3]
                    frame["dihedrals"]["k"] = dihedrals[:, 4]
                    frame["dihedrals"]["l"] = dihedrals[:, 5]

        if globals().get("masses"):
            frame["atoms"]["mass"] = np.array(
                [masses[i] for i in frame["atoms"]["type"]]
            )

        frame.box = mp.Box(
            np.diag(
                [
                    box["xhi"] - box["xlo"],
                    box["yhi"] - box["ylo"],
                    box["zhi"] - box["zlo"],
                ]
            )
        )

        return frame


class LammpsDataSaver:

    def __init__(self, frame: mp.Frame, file: str | Path, atom_style: str = "full"):
        self._frame = frame
        self._file = Path(file)
        self._atom_style = atom_style

    def save(self):

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
            f.write(f"{frame['global_props']['n_atoms']} atoms\n")
            f.write(f"{frame['global_props']['n_bonds']} bonds\n")
            f.write(f"{frame['global_props']['n_angles']} angles\n")
            f.write(f"{frame['global_props']['n_dihedrals']} dihedrals\n\n")
            f.write(f"{frame['global_props']['n_atomtypes']} atom types\n")
            f.write(f"{frame['global_props']['n_bondtypes']} bond types\n")
            f.write(f"{frame['global_props']['n_angletypes']} angle types\n")
            f.write(f"{frame['global_props']['n_dihedraltypes']} dihedral types\n\n")

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
