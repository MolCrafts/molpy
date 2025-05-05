from pathlib import Path

from .base import DataWriter, DataReader
import numpy as np

import molpy as mp
from collections import defaultdict
from nesteddict import ArrayDict


class PDBReader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(path=file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    def read(self, frame):

        with open(self._path, "r") as f:

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
                    atoms["x"].append(float(line[30:38]))
                    atoms["y"].append(float(line[38:46]))
                    atoms["z"].append(float(line[46:54]))
                    atoms["element"].append(line[76:78].strip())

                elif line.startswith("CONECT"):

                    bond_indices = list(map(lambda l: int(l), line.split()[1:]))
                    i = bond_indices[0]
                    for j in bond_indices[1:]:
                        bonds.append([i, j] if i < j else [j, i])

            bonds = np.unique(np.array(bonds), axis=0)

            if len(set(atoms["name"])) != len(atoms["name"]):
                atom_name_counter = defaultdict(int)
                for i, name in enumerate(atoms["name"]):
                    atom_name_counter[name] += 1
                    if atom_name_counter[name] > 1:
                        atoms["name"][i] = f"{name}{atom_name_counter[name]}"

            frame["atoms"] = ArrayDict.from_dicts(atoms)
            frame.box = mp.Box()

            if len(bonds):
                frame["bonds"] = ArrayDict.from_dicts(
                    {
                        "i": bonds[:, 0],
                        "j": bonds[:, 1],
                    }
                )

            return frame


class PDBWriter(DataWriter):

    def __init__(
        self,
        path: str | Path,
    ):
        super().__init__(path=path)

    def write(self, frame):
        def as_builtin(x):
            if isinstance(x, np.ndarray):
                if x.shape == ():  # 0-d
                    return x.item()
                elif x.shape == (1,):
                    return x[0].item()
            return x
        with open(self.path, "w") as f:
            atoms = frame["atoms"]
            for i, atom in enumerate(atoms.iterrows()):
                serial = as_builtin(atom.get("id", i))
                altLoc = as_builtin(atom.get("altLoc", ""))
                unique_name = as_builtin(atom.get("unique_name", atom.get("name", "UNK")))
                resName = as_builtin(atom.get("resName", "UNK"))
                chainID = as_builtin(atom.get("chainID", "A"))
                resSeq = as_builtin(atom.get("resSeq", atom.get("molid", 1)))
                iCode = as_builtin(atom.get("iCode", ""))
                elem = as_builtin(atom.get("element", "X"))
                x, y, z = map(as_builtin, atom["xyz"])

                f.write(
                    f"{'ATOM':6s}{serial:>5d} {unique_name.upper():>4s}{altLoc:1s}{resName:>3s} {chainID:1s}{resSeq:>4d}{iCode:1s}   "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{' '*22}{elem:>2s}  \n"
                )

            bonds = defaultdict(list)
            if "bonds" in frame:
                for bond in frame["bonds"].iterrows():
                    i = as_builtin(bond["i"])
                    j = as_builtin(bond["j"])
                    bonds[i].append(j)
                    bonds[j].append(i)

                for i, js in bonds.items():
                    if len(js) > 4:
                        raise ValueError(
                            f"PDB only supports up to 4 bonds, but atom {i} has {len(js)} bonds: {js}"
                        )
                    f.write(f"{'CONECT':6s}{i:>5d}{''.join([f'{j:>5d}' for j in js])}\n")

            f.write("END\n")