from pathlib import Path

import pandas as pd

from molpy import Element

from .base import DataReader


class TopReader(DataReader):
    def __init__(self, file: str | Path):
        super().__init__(file)
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    def read(self, frame=None):
        with open(self._file) as f:
            lines = f.readlines()

        self.atoms = []
        self.bonds = []

        section = None
        for line in map(TopReader.sanitizer, lines):
            if line.startswith("["):
                section = line
                continue

            if section == "[ atoms ]":
                self._parse_atom_section(line)

            if section == "[ bonds ]":
                self._parse_bond_section(line)

            if section == "[ pairs ]":
                self._parse_pair_section(line)

            if section == "[ angles ]":
                self._parse_angle_section(line)

            if section == "[ dihedrals ]":
                self._parse_dihedral_section(line)

        frame["atoms"] = pd.DataFrame.from_records(
            self.atoms,
        )
        frame["bonds"] = pd.DataFrame.from_records(
            self.bonds, columns=["id", "i", "j", "type"]
        )
        frame["pairs"] = pd.DataFrame.from_records(
            self.pairs, columns=["id", "i", "j", "type"]
        )
        frame["angles"] = pd.DataFrame.from_records(
            self.angles, columns=["id", "i", "j", "k", "type"]
        )
        frame["dihedrals"] = pd.DataFrame.from_records(
            self.dihedrals, columns=["id", "i", "j", "k", "l", "type"]
        )

        self.assign_atomic_numbers(self.atoms, None)

        return frame

    def assign_atomic_numbers(self, atoms, restemp):
        for atom in atoms:
            atomic_number = self._guess_atomic_number(atom["name"], restemp).number
            if atomic_number == 0:
                atomic_number = self._guess_atomic_number(atom["type"], restemp).number
            atom["number"] = atomic_number

    def _guess_atomic_number(self, name, residue=None):
        """Guesses the atomic number"""
        # Special-case single-atom residues, which are almost always ions
        name = "".join(c for c in name if c.isalpha())
        if (residue is None or len(residue.atoms) == 1) and len(name) > 1:
            try:
                return Element(name[0].upper() + name[1].lower())
            except KeyError:
                return Element(name)
        return Element(name)

    def _parse_atom_section(self, line):
        data = line.split()
        index = data[0]
        type = data[1]
        data[2]
        data[3]
        name = data[4]
        data[5]
        data[6]

        x = float(data[3])
        y = float(data[4])
        z = float(data[5])

        vx = vy = vz = None
        if len(data) >= 9:
            vx = float(data[6])
            vy = float(data[7])
            vz = float(data[8])

        self.atoms.append(
            {
                "id": index,
                "name": name,
                "type": type,
                "x": x,
                "y": y,
                "z": z,
                "vx": vx,
                "vy": vy,
                "vz": vz,
            }
        )
