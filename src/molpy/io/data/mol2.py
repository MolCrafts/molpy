from pathlib import Path

from molpy import Element
import numpy as np

import molpy as mp
from collections import defaultdict
import pandas as pd
from .base import DataReader, DataWriter


class Mol2Reader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(file)
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    def read(self, frame=None):

        with open(self._file, "r") as f:
            lines = f.readlines()

        self.atoms = []
        self.bonds = []

        section = None
        for line in map(Mol2Reader.sanitizer, lines):
            if line.startswith("@<TRIPOS>"):
                section = line[9:]
                continue

            if section == "MOLECULE":
                self._parse_molecule_section(line)

            elif section == "ATOM":
                self._parse_atom_section(line)

            elif section == "BOND":
                self._parse_bond_section(line)

        self.assign_atomic_numbers(self.atoms, None)

        frame["atoms"] = pd.DataFrame.from_records(
            self.atoms,
        )
        frame["bonds"] = pd.DataFrame.from_records(
            self.bonds, columns=["id", "i", "j", "type"]
        )

        return frame

    def assign_atomic_numbers(self, atoms, restemp):
        for atom in atoms:
            atomic_number = self._guess_atomic_number(atom['name'], restemp).number
            if atomic_number == 0:
                atomic_number = self._guess_atomic_number(atom['type'], restemp).number
            atom['number'] = atomic_number

    def _guess_atomic_number(self, name, residue=None):
        """Guesses the atomic number"""
        # Special-case single-atom residues, which are almost always ions
        name = "".join(c for c in name if c.isalpha())
        if residue is None or len(residue.atoms) == 1:
            if len(name) > 1:
                try:
                    return Element(name[0].upper() + name[1].lower())
                except KeyError:
                    return Element(name)
        return Element(name)

    def _parse_molecule_section(self, line):
        pass

    def _parse_atom_section(self, line):
        """Parse an atom entry in the mol2 file."""
        data = line.split()
        index = int(data[0])
        name = data[1]
        x, y, z = map(float, data[2:5])
        atom_type = data[5]
        subst_id = int(data[6]) if len(data) > 6 else 0
        subst_name = data[7] if len(data) > 7 else ""
        charge = float(data[8]) if len(data) > 8 else 0.0
        self.atoms.append(
            {
                "id": index,
                "name": name,
                "x": x,
                "y": y,
                "z": z,
                "type": atom_type,
                "subst_id": subst_id,
                "subst_name": subst_name,
                "charge": charge
            }
        )

    def _parse_bond_section(self, line):
        """Parse a bond entry in the mol2 file."""
        data = line.split()
        index = int(data[0])
        atom1 = int(data[1])
        atom2 = int(data[2])
        bond_type = data[3]
        self.bonds.append((index, atom1, atom2, bond_type))
