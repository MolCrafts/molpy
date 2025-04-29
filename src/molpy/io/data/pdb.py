from pathlib import Path

from .base import DataWriter, DataReader
import numpy as np

import molpy as mp
from collections import defaultdict
import pandas as pd

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

            if len(set(atoms['name'])) != len(atoms['name']):
                atom_name_counter = defaultdict(int)
                for i, name in enumerate(atoms['name']):
                    atom_name_counter[name] += 1
                    if atom_name_counter[name] > 1:
                        atoms['name'][i] = f"{name}{atom_name_counter[name]}"

            frame["atoms"] = pd.DataFrame(
                atoms
            )
            frame.box = mp.Box()
            
            if len(bonds):
                frame["bonds"] = pd.DataFrame(
                    {
                        "i": bonds[:, 0],
                        "j": bonds[:, 1],
                    }
                )

            return frame

class PDBWriter(DataWriter):

    def __init__(self, path: str | Path, ):
        super().__init__(path=path)

    def write(self, frame):

        with open(self._path, "w") as f:

            for i, atom in enumerate(frame["atoms"].iterrows()):
                serial = atom.get("id", i)
                altLoc = atom.get("altLoc", "")
                unique_name = atom.get("unique_name", atom.get("name", "UNK"))
                # if name in atom_name_remap:
                #     unique_name = name + str(atom_name_remap[name])
                #     atom_name_remap[name] += 1
                # else:
                #     unique_name = name
                #     atom_name_remap[name] = 1
                resName = atom.get("resName", "UNK")
                chainID = atom.get("chainID", "A")
                resSeq = atom.get("resSeq", atom.get("molid", 1))
                iCode = atom.get("iCode", "")
                elem = atom.get("element", "X")
                x, y, z = atom["xyz"].T

                f.write(
                    f"{'ATOM':6s}{serial:>5d}{' '*1}{unique_name.upper():>4s}{altLoc:1s}{resName:>3s}{' '*1}{chainID:1s}{resSeq:>4d}{iCode:1s}{' '*3}{x:>8.3f}{y:>8.3f}{z:>8.3f}{' '*22}{elem:>2s}  \n"
                )

            bonds = defaultdict(list)
            if "bonds" in frame:
                for bond in frame["bonds"].iterrows():
                    bonds[bond["i"]].append(bond["j"])
                    bonds[bond["j"]].append(bond["i"])

                for i, js in bonds.items():
                    assert len(js) <= 4, ValueError(f"PDB only supports up to 4 bonds, but atom {i} has {len(js)} bonds, which are {js}")
                    f.write(f"{'CONECT':6s}{i:>5d}{''.join([f'{j:>5d}' for j in js])}\n")

            f.write("END\n")
