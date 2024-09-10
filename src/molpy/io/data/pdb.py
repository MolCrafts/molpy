from pathlib import Path

import numpy as np

import molpy as mp
from collections import defaultdict


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
                "xyz": [],
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
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )

                if line.startswith("CONECT"):

                    bond_indices = list(map(lambda l: int(l), line.split()[1:]))
                    i = bond_indices[0]
                    for j in bond_indices[1:]:
                        bonds.append([i, j] if i < j else [j, i])

            bonds = np.unique(np.array(bonds), axis=0)

            frame["bonds"]["i"] = bonds[:, 0]
            frame["bonds"]["j"] = bonds[:, 1]
            for key, value in atoms.items():
                frame["atoms"][key] = np.array(value)

            return frame


class PDBWriter:

    def __init__(self, frame: mp.Frame, file: str | Path):
        self._frame = frame
        self._file = Path(file)

    def write(self):

        with open(self._file, "w") as f:

            for i, atom in enumerate(self._frame["atoms"]):
                f.write(
                    f"{'ATOM':6s}{atom['id']:>5d} {atom['name']:4s} {atom['resName']:>3s} {atom['chainID']:>1s}{atom['resSeq']:>4d}    {atom['xyz'][0]:>8.3f}{atom['xyz'][1]:>8.3f}{atom['xyz'][2]:>8.3f}{" "*26}\n"
                )

            bonds = defaultdict(list)
            for i, j in zip(self._frame["bonds"]["i"], self._frame["bonds"]["j"]):
                bonds[i].append(j)

            for i, js in bonds.items():
                assert len(js) <= 4
                f.write(f"{'CONECT':6s}{i:>5d}{''.join([f'{j:>5d}' for j in js])}\n")
