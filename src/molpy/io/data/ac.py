from pathlib import Path
from molpy import Element
from .base import DataReader
from molpy.core.frame import _dict_to_dataset
import molpy as mp

class AcReader(DataReader):

    def __init__(self, file: str | Path):
        super().__init__(Path(file))
        self._file = Path(file)

    def read(self, frame: mp.Frame) -> mp.Frame:
        with open(self._file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.atoms = []
        self.bonds = []

        for line in lines:
            if line.startswith("ATOM"):
                self._parse_atom_section(line)
            elif line.startswith("BOND"):
                self._parse_bond_section(line)

        if self.atoms:
            self.assign_atomic_numbers(self.atoms)
            keys = self.atoms[0].keys()
            frame["atoms"] = _dict_to_dataset({k: [d[k] for d in self.atoms] for k in keys})
        
        if self.bonds:
            keys = self.bonds[0].keys()
            frame["bonds"] = _dict_to_dataset({k: [d[k] for d in self.bonds] for k in keys})

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
            "q": charge,
            "type": atom_type
        })

    def _parse_bond_section(self, line):
        # Example:
        # BOND    1    1    2    1      C   H1
        tokens = line.split()
        bond_id = int(tokens[1])
        atom1 = int(tokens[2])
        atom2 = int(tokens[3])
        bond_type = int(tokens[4])  # usually just 1 (single)

        self.bonds.append({
            "id": bond_id,
            "i": atom1,
            "j": atom2,
            "type": bond_type
        })

    def assign_atomic_numbers(self, atoms):
        for atom in atoms:
            element_data = self._guess_atomic_number(atom["name"])
            atomic_number = element_data.number
            if atomic_number == 0:
                element_data = self._guess_atomic_number(atom["type"])
                atomic_number = element_data.number
            atom["number"] = atomic_number

    def _guess_atomic_number(self, name):
        name = ''.join([c for c in name if c.isalpha()])
        try:
            return Element(name.capitalize())
        except KeyError:
            return Element(0)
