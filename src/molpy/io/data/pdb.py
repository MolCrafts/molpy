from pathlib import Path

from .base import DataWriter, DataReader
import numpy as np

import molpy as mp
from collections import defaultdict
from nesteddict import ArrayDict


class PDBReader(DataReader):

    def __init__(self, file: Path):
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
                "xyz": [],
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
                    atoms["xyz"].append(
                        (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                    )
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

            frame["atoms"] = ArrayDict(atoms)
            frame["box"] = mp.Box()

            if len(bonds):
                frame["bonds"] = ArrayDict(
                    {
                        "i": bonds[:, 0],
                        "j": bonds[:, 1],
                    }
                )


            return frame


class PDBWriter(DataWriter):

    def __init__(
        self,
        path: Path,
    ):
        super().__init__(path=path)

    def write(self, frame):

        with open(self._path, "w") as f:
            atoms = frame["atoms"]
            name = frame.get("name", "MOL")
            f.write(f"REMARK  {name}\n")
            for i, atom in enumerate(atoms.iterrows()):
                # serial = as_builtin(atom.get("id", i))
                serial = i + 1
                altLoc = atom.get("altLoc", np.array("")).item()
                unique_name = atom.get("unique_name", atom.get("name", np.array("UNK"))).item()

                resName = atom.get("resName", np.array("UNK")).item()
                chainID = atom.get("chainID", np.array("A")).item()
                resSeq = atom.get("resSeq", atom.get("molid", np.array(1))).item()
                iCode = atom.get("iCode", np.array("")).item()
                elem = atom.get("element", np.array("X")).item()
                x, y, z = atom["xyz"]

                f.write(
                    f"{'HETATM':6s}{serial:>5d} {unique_name.upper():>4s}{altLoc:1s}{resName:>3s} {chainID:1s}{resSeq:>4d}{iCode:1s}   "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{' '*22}{elem:>2s}  \n"
                )

            bonds = defaultdict(list)
            if "bonds" in frame:
                for bond in frame["bonds"].iterrows():
                    i = int(bond["i"] + 1)
                    j = int(bond["j"] + 1)
                    bonds[i].append(j)
                    bonds[j].append(i)

                for i, js in bonds.items():
                    if len(js) > 4:
                        raise ValueError(
                            f"PDB only supports up to 4 bonds, but atom {i} has {len(js)} bonds: {js}"
                        )
                    f.write(
                        f"{'CONECT':6s}{i:>5d}{''.join([f'{j:>5d}' for j in js])}\n"
                    )

            f.write("END\n")
