from pathlib import Path

import numpy as np

import molpy as mp
from collections import defaultdict
import pyarrow as pa


class PDBReader:

    def __init__(self, file: str | Path):
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    def read(self):

        with open(self._file, "r") as f:

            frame = mp.Frame()

            lines = filter(
                lambda line: line.startswith("ATOM")
                or line.startswith("CONECT")
                or line.startswith("HETATM"),
                map(PDBReader.sanitizer, f),
            )

            atoms = {
                "id": [],
                "name": [],
                "resName": [],
                "chainID": [],
                "resSeq": [],
                "x": [],
                "y": [],
                "z": [],
                "element": [],
            }

            bonds = []

            for line in lines:

                if line.startswith("ATOM") or line.startswith("HETATM"):

                    atoms["id"].append(int(line[6:11]))
                    atoms["name"].append(line[12:16].strip())
                    atoms["resName"].append(line[17:20].strip())
                    atoms["chainID"].append(line[21])
                    atoms["resSeq"].append(int(line[22:26]))
                    atoms["x"].append(
                        float(line[30:38])
                    )
                    atoms['y'].append(float(line[38:46]))
                    atoms['z'].append(float(line[46:54]))
                    atoms["element"].append(line[76:78].strip())

                elif line.startswith("CONECT"):

                    bond_indices = list(map(lambda l: int(l), line.split()[1:]))
                    i = bond_indices[0]
                    for j in bond_indices[1:]:
                        bonds.append([i, j] if i < j else [j, i])

            bonds = np.unique(np.array(bonds), axis=0)

            frame["atoms"] = pa.table(
                atoms
            )
            frame["bonds"] = pa.table(
                {
                    "i": pa.array(bonds[:, 0]),
                    "j": pa.array(bonds[:, 1]),
                }
            )

            return mp.System(
                box=mp.Box(),
                ff=mp.ForceField(),
                frame=frame,
            )


class PDBWriter:

    def __init__(self, file: str | Path):
        self._file = Path(file)

    def write(self, system):

        frame = system.frame

        with open(self._file, "w") as f:

            atom_name_remap = {}

            for atom in frame["atoms"].to_pylist():
                serial = atom["id"]
                altLoc = atom.get("altLoc", "")
                name = atom["name"]
                if name in atom_name_remap:
                    unique_name = name + str(atom_name_remap[name])
                    atom_name_remap[name] += 1
                else:
                    unique_name = name
                    atom_name_remap[name] = 1
                resName = atom.get("resName", "UNK")
                chainID = atom.get("chainID", "")
                resSeq = atom.get("resSeq", atom.get("molid", 1))
                iCode = atom.get("iCode", "")
                x = atom["x"]
                y = atom["y"]
                z = atom["z"]
                f.write(
                    f"{'ATOM':6s}{serial:>5d}{' '*1}{unique_name.upper():>4s}{altLoc:1s}{resName:>3s}{' '*1}{chainID:1s}{resSeq:>4d}{iCode:1s}{' '*3}{x:>8.3f}{y:>8.3f}{z:>8.3f}{" "*26}\n"
                )

            bonds = defaultdict(list)
            for bond in frame["bonds"].to_pylist():
                bonds[bond["i"]].append(bond["j"])
                bonds[bond["j"]].append(bond["i"])

            for i, js in bonds.items():
                assert len(js) <= 4
                f.write(f"{'CONECT':6s}{i:>5d}{''.join([f'{j:>5d}' for j in js])}\n")

            f.write("END\n")
