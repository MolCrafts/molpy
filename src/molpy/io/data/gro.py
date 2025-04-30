from .base import DataReader
from pathlib import Path
import pandas as pd
from molpy import Element


class GroReader(DataReader):
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

        lines = list(map(GroReader.sanitizer, lines))

        natoms = int(lines[1])

        for line in lines[2:2+natoms]:
            self._parse_atom_section(line)

        self.assign_atomic_numbers(self.atoms, None)

        frame["atoms"] = pd.DataFrame.from_records(
            self.atoms,
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
    
    def _parse_atom_section(self, line):
        data = line.split()
        subst_name = data[0]
        name = data[1]
        index = int(data[2])

        x = float(data[3])
        y = float(data[4])
        z = float(data[5])

        vx = vy = vz = None
        if len(data) >= 9:
            vx = float(data[6])
            vy = float(data[7])
            vz = float(data[8])

        self.atoms.append({
            "id": index,
            "name": name,
            "subst_name": subst_name,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz
        })