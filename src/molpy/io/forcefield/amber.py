from pathlib import Path
import molpy as mp
from itertools import islice
from typing import Iterator

class AmberForceFieldReader:

    def __init__(
        self, file: str | Path, forcefield: mp.ForceField | None = None
    ):
        
        self.file = file
        if forcefield is None:
            self.forcefield = mp.ForceField()
        else:
            self.forcefield = forcefield

    @staticmethod
    def sanitizer(line: str) -> str:

        return line.strip()
    
    def get_section(self, lines: Iterator[str]) -> str:
        section = []
        for line in lines:
            if line.startswith('%'):
                field = line.split()[1]
                break
            section.append(line)
        return section, field

    def read(self, system: mp.System) -> mp.ForceField:

        with open(self.file, "r") as f:
            
            lines = filter(lambda line: line, map(AmberForceFieldReader.sanitizer, f.readlines()))

        atoms = {}

        for line in lines:

            if line.startswith(f'%FLAG'):
                field = line.split()[1]
                match field:

                    case 'TITLE':
                        pass

                    case "POINTERS":
                        pointer_lines, field = self.get_section(lines)
                        meta = self._read_pointers(pointer_lines)
                        continue

                    case "ATOM_NAME":
                        atom_names, field = self.get_section(lines)
                        atoms['name'] = self._read_atom_name(atom_names)
                        continue

                    case "CHARGE":
                        charges, field = self.get_section(lines)
                        atoms['charge'] = self._read_charge(charges)
                        continue

                    case "ATOM_NUMBER":
                        atom_numbers, field = self.get_section(lines)
                        atoms['element'] = self._read_atom_number(atom_numbers)
                        continue

                    case "MASS":
                        masses, field = self.get_section(lines)
                        atoms['mass'] = self._read_mass(masses)
                        continue

                    case "ATOM_TYPE_INDEX":
                        atom_types, field = self.get_section(lines)
                        atoms['type'] = self._read_atom_type(lines)
                        continue

                    case "RESIDUE_LABEL":
                        self._read_residues(lines)

    def _read_pointers(self, lines):
        meta_fields = (
            "NATOM",    "NTYPES", "NBONH",  "MBONA",  "NTHETH", "MTHETA",
               "NPHIH",    "MPHIA",  "NHPARM", "NPARM",  "NNB",    'NRES',
               "NBONA",    "NTHETA", "NPHIA",  "NUMBND", "NUMANG", "NPTRA",
               "NATYP",    "NPHB",   "IFPERT", "NBPER",  "NGPER",  "NDPER",
               "MBPER",    "MGPER",  "MDPER",  "IFBOX",  "NMXRS",  "IFCAP",
               "NUMEXTRA", "NCOPY"
        )
        meta_data = dict(zip(meta_fields, map(int, next(lines).split())))
        return meta_data
    
    def _read_atom_name(self, lines: Iterator[str]):
        
        names = []

        for line in lines:
            if line.startswith('%'):
                break
            names.extend(line[i: i+4].strip() for i in range(0, len(line), 4))

        return names

    def _read_charge(self, lines: Iterator[str]):

        charges = []
        for line in lines:
            if line.startswith('%'):
                break
            charges.extend(map(float, line.split()))

        return charges

    def _read_atom_number(self, lines: Iterator[str]):

        atom_numbers = []
        for line in lines:
            if line.startswith('%'):
                break
            atom_numbers.extend(map(int, line.split()))

        return atom_numbers

    def _read_mass(self, lines: Iterator[str]):

        masses = []
        for line in lines:
            if line.startswith('%'):
                break
            masses.extend(map(float, line.split()))

        return masses
    
    def _read_atom_type(self, lines: Iterator[str]):

        atom_types = []
        for line in lines:
            if line.startswith('%'):
                break
            atom_types.extend(map(int, line.split()))

        return atom_types
    
    def _read_residues(self, lines: Iterator[str]):

        residues = []

        return residues
    
    def _read_residue(self, lines: Iterator[str]):
        next(lines)
        name = next(lines).strip()
        
        for line in lines:
            if line.startswith(f'%FLAG'):
                field = line.split()[1]

                match field:

                    case 'RESIDUE_POINTER':
                        break

                    case 'BOND_FORCE_CONSTANT':
                        bfc, field = self.get_section(lines)
                        self._read_residue_bond_force_constant(lines)

                    case 'BOND_EQUIL_VALUE':
                        bev, field = self.get_section(lines)
                        self._read_residue_bond_equil_value(lines)

    def _read_residue_bond_force_constant(self, lines: Iterator[str]):

        bond_force_constants = []
        for line in lines:
            if line.startswith('%'):
                break
            bond_force_constants.extend(map(float, line.split()))

        return bond_force_constants
    
    def _read_residue_bond_equil_value(self, lines: Iterator[str]):
            
        bond_equil_values = []
        for line in lines:
            if line.startswith('%'):
                break
            bond_equil_values.extend(map(float, line.split()))

        return bond_equil_values

