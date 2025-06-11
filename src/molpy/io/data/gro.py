from molpy.core.frame import _dict_to_dataset
from .base import DataReader
from pathlib import Path
from molpy import Element


class GroReader(DataReader):
    def __init__(self, file: str | Path):
        super().__init__(file)
        self._file = Path(file)

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.rstrip()

    def read(self, frame):
        with open(self._file, "r") as f:
            lines = f.readlines()

        lines = list(map(GroReader.sanitizer, lines))

        self._parse_title(lines[0], frame)
        natoms = int(lines[1])
        self._parse_atom_section(lines[2 : 2 + natoms], frame)

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
        if residue is None or len(residue.atoms) == 1:
            if len(name) > 1:
                try:
                    return Element(name[0].upper() + name[1].lower())
                except KeyError:
                    return Element(name)
        return Element(name)

    def _parse_title(self, line, frame):
        comment, _, timestep = line.partition("t=")
        if timestep:
            frame.timestep = int(timestep.strip())

    def _parse_atom_section(self, line, frame):
        atoms = []
        for line in line:
            atom = {
                "res_number": line[0:5].strip(),
                "res_name": line[5:10].strip(),
                "name": line[10:15].strip(),
                "atomic_number": int(line[15:20].strip()),
                "xyz": (float(line[20:28]), float(line[28:36]), float(line[36:44])),
            }

            if len(line) > 44:
                atom["vx"] = float(line[44:53])
                atom["vy"] = float(line[53:62])
                atom["vz"] = float(line[62:71])

            atoms.append(atom)
        if atoms:
            keys = atoms[0].keys()
            frame["atoms"] = _dict_to_dataset({k: [d[k] for d in atoms] for k in keys})
