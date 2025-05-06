from pathlib import Path
from molpy import Element
from .base import DataReader
from nesteddict import ArrayDict

class AcReader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(file)
        self._file = Path(file)

    def read(self, frame=None):
        with open(self._file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.atoms = []
        self.bonds = []

        for line in lines:
            if line.startswith("ATOM"):
                self._parse_atom_section(line)
            elif line.startswith("BOND"):
                self._parse_bond_section(line)

        self.assign_atomic_numbers(self.atoms)

        frame["atoms"] = ArrayDict.from_dicts(self.atoms)
        frame["bonds"] = ArrayDict.from_dicts(self.bonds)

        return frame

    def _parse_atom_section(self, line):
        # Example:
        # ATOM      1  C   UNK     1       0.000   0.000   0.000 -0.094100        c3
        tokens = line.split()
        atom_id = int(tokens[1])
        name = tokens[2]
        resname = tokens[3]
        res_id = int(tokens[4])
        xyz = tuple(map(float, tokens[5:8]))
        charge = float(tokens[8])
        atom_type = tokens[9]

        self.atoms.append({
            "id": atom_id,
            "name": name,
            "resName": resname,
            "resSeq": res_id,
            "xyz": xyz,
            "charge": charge,
            "type": atom_type
        })

    def _parse_bond_section(self, line):
        # Example:
        # BOND    1    1    2    1      C   H1
        tokens = line.split()
        bond_id = int(tokens[1])
        atom1 = int(tokens[2])
        atom2 = int(tokens[3])
        bond_type = tokens[4]  # usually just 1 (single)

        self.bonds.append({
            "id": bond_id,
            "i": atom1,
            "j": atom2,
            "type": bond_type
        })

    def assign_atomic_numbers(self, atoms):
        for atom in atoms:
            atomic_number = self._guess_atomic_number(atom["name"]).number
            if atomic_number == 0:
                atomic_number = self._guess_atomic_number(atom["type"]).number
            atom["number"] = atomic_number

    def _guess_atomic_number(self, name):
        name = ''.join([c for c in name if c.isalpha()])
        try:
            return Element(name.capitalize())
        except KeyError:
            return Element("X")
