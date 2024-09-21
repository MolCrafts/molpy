from pathlib import Path
import molpy as mp
from typing import Iterator
import pyarrow as pa

class AmberPrmtopReader:

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

    def read(self, system: mp.System) -> mp.ForceField:

        with open(self.file, "r") as f:
            
            lines = filter(lambda line: line, map(AmberPrmtopReader.sanitizer, f.readlines()))

        raw_data = {        }
        data = []
        flag = None

        for line in lines:

            if line.startswith(f'%FLAG'):
                if flag:
                    if flag in raw_data:
                        raw_data[flag].extend(data)
                    else:
                        raw_data[flag] = data

                flag = line.split()[1]
                data = []

            elif line.startswith(f'%FORMAT'):
                pass

            else:
                data.extend(line.split())

        if flag:
            raw_data[flag] = data

        atoms = {}
        bonds = {}
        angles = {}
        dihedrals = {}

        for key, value in raw_data.items():
            match key:

                case 'TITLE':
                    title = value[0]

                case 'POINTERS':
                    system.frame['props'] = self._read_pointers(raw_data[key])

                case 'ATOM_NAME':
                    atoms['name'] = self._read_atom_name(iter(value))

                case 'CHARGE':
                    atoms['charge'] = self._read_charge(iter(value))

                case 'ATOMIC_NUMBER':
                    atoms['element'] = self._read_atom_number(iter(value))

                case 'MASS':
                    atoms['mass'] = self._read_mass(iter(value))

                case 'ATOM_TYPE_INDEX':
                    atoms['type'] = self._read_atom_type(iter(value))

                # case 'NUMBER_EXCLUDED_ATOMS':
                case 'NONBONDED_PARM_INDEX':
                    nonbonded_index, nonbonded_params, acoeff, bcoeff = self.parse_nonbonded_params(raw_data)

                # case 'RESIDUE_LABEL':
                case 'BOND_FORCE_CONSTANT':
                    bonds['force_constant'] = list(map(float, value))

                case 'BOND_EQUIL_VALUE':
                    bonds['equil_value'] = list(map(float, value))

                case 'ANGLE_FORCE_CONSTANT':
                    angles['force_constant'] = list(map(float, value))
                
                case 'ANGLE_EQUIL_VALUE':
                    angles['equil_value'] = list(map(float, value))

                case 'DIHEDRAL_FORCE_CONSTANT':
                    dihedrals['force_constant'] = list(map(float, value))

                case 'DIHEDRAL_PERIODICITY':
                    dihedrals['periodicity'] = list(map(float, value))

                case 'DIHEDRAL_PHASE':
                    dihedrals['phase'] = list(map(float, value))

                # case 'SCEE_SCALE_FACTOR':
                # case 'SCNB_SCALE_FACTOR':

        atoms['id'] = range(1, system.frame['props']['NATOM'] + 1)
        bonds['id'] = range(1, system.frame['props']['NUMBND'] + 1)
        angles['id'] = range(1, system.frame['props']['NUMANG'] + 1)
        dihedrals['id'] = range(1, system.frame['props']['NPTRA'] + 1)

        system.frame['atoms'] = pa.table(atoms)
        system.frame['bonds'] = pa.table(bonds)
        system.frame['angles'] = pa.table(angles)
        system.frame['dihedrals'] = pa.table(dihedrals)

        return system

    def _read_pointers(self, lines):
        meta_fields = (
            "NATOM",    "NTYPES", "NBONH",  "MBONA",  "NTHETH", "MTHETA",
               "NPHIH",    "MPHIA",  "NHPARM", "NPARM",  "NNB",    'NRES',
               "NBONA",    "NTHETA", "NPHIA",  "NUMBND", "NUMANG", "NPTRA",
               "NATYP",    "NPHB",   "IFPERT", "NBPER",  "NGPER",  "NDPER",
               "MBPER",    "MGPER",  "MDPER",  "IFBOX",  "NMXRS",  "IFCAP",
               "NUMEXTRA", "NCOPY"
        )
        meta_data = dict(zip(meta_fields, map(int, ' '.join(lines).split())))
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
        for line in lines:
            if line.startswith(f'%FLAG RESIDUE_LABEL'):
                self._read_residue(lines)

        return residues
    
    def _read_residue(self, lines: Iterator[str]):

        next(lines)
        next(lines)
        force_constants = {}
        equil_values = {}
        scale_factors = {}
        lj_params = {}
        bonds = {
            'i': [],
            'j': [],
            'id': []
        }
        angles = {
            'i': [],
            'j': [],
            'k': [],
            'id': []
        }
        dihedrals = {
            'i': [],
            'j': [],
            'k': [],
            'l': [],
            'id': []
        }
        
        for line in lines:
            if line.startswith(f'%FLAG'):
                field = line.split()[1]

                match field:

                    case 'RESIDUE_POINTER':
                        break

                    case 'BOND_FORCE_CONSTANT' | 'ANGLE_FORCE_CONSTANT' | 'DIHEDRAL_FORCE_CONSTANT':
                        _, field = self.get_section(lines)
                        force_constants[field] = self._read_section(lines)

                    case 'BOND_EQUIL_VALUE' | 'ANGLE_EQUIL_VALUE' | 'DIHEDRAL_EQUIL_VALUE':
                        _, field = self.get_section(lines)
                        equil_values[field] = self._read_section(lines)

                    case 'DIHEDRAL_PERIODICITY':
                        _, field = self.get_section(lines)
                        dihedral_periodicity = self._read_section(lines)

                    case 'DIHEDRAL_PHASE':
                        _, field = self.get_section(lines)
                        dihedral_phase = self._read_section(lines)

                    case 'SCEE_SCALE_FACTOR' | 'SCNB_SCALE_FACTOR':
                        _, field = self.get_section(lines)
                        scale_factors[field] = self._read_section(lines)

                    case 'LENNARD_JONES_ACOEF' | 'LENNARD_JONES_BCOEF':
                        _, field = self.get_section(lines)
                        lj_params[field] = self._read_section(lines)

                    case 'BONDS_INC_HYDROGEN' | 'BONDS_WITHOUT_HYDROGEN':
                        _, field = self.get_section(lines)
                        bond_info = self._read_section(lines)
                        bonds['i'].extend(bond_info[0::3])
                        bonds['j'].extend(bond_info[1::3])
                        bonds['id'].extend(bond_info[2::3])

                    case 'ANGLES_INC_HYDROGEN' | 'ANGLES_WITHOUT_HYDROGEN':
                        _, field = self.get_section(lines)
                        angle_info = self._read_section(lines)
                        angles['i'].extend(angle_info[0::4])
                        angles['j'].extend(angle_info[1::4])
                        angles['k'].extend(angle_info[2::4])
                        angles['id'].extend(angle_info[3::4])

                    case 'DIHEDRALS_INC_HYDROGEN' | 'DIHEDRALS_WITHOUT_HYDROGEN':
                        _, field = self.get_section(lines)
                        dihedral_info = self._read_section(lines)
                        dihedrals['i'].extend(dihedral_info[0::5])
                        dihedrals['j'].extend(dihedral_info[1::5])
                        dihedrals['k'].extend(dihedral_info[2::5])
                        dihedrals['l'].extend(dihedral_info[3::5])
                        dihedrals['id'].extend(dihedral_info[4::5])

                    case 'AMBER_ATOM_TYPE':
                        _, field = self.get_section(lines)
                        type_name = self._read_section(lines)

    def parse_nonbonded_params(self, raw_data: dict):
        nonbonded_params = {}
        nonbonded_index = [int(x) for x in raw_data['NONBONDED_PARM_INDEX']]
        acoef = [float(x) for x in raw_data['LENNARD_JONES_ACOEF']]
        bcoef = [float(x) for x in raw_data['LENNARD_JONES_BCOEF']]
        
        for i in range(len(nonbonded_index)):
            index = nonbonded_index[i] - 1 
            if index >= 0:
                a_param = acoef[index]
                b_param = bcoef[index]
                nonbonded_params[i + 1] = (a_param, b_param)

        return nonbonded_index, nonbonded_params, acoef, bcoef