from pathlib import Path
import molpy as mp
import numpy as np
from typing import Iterator
import pyarrow as pa
from itertools import accumulate

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
                data.append(line)

        if flag:
            raw_data[flag] = data

        atoms = {}
        bonds = {
            'type': [],
            'i': [],
            'j': [],
        }
        angles = {
            'type': [],
            'i': [],
            'j': [],
            'k': [],
        }
        dihedrals = {
            'type': [],
            'i': [],
            'j': [],
            'k': [],
            'l': [],
        }
        pairs = {}

        bond_params = {}
        angle_params = {}
        dihedral_params = {}

        for key, value in raw_data.items():
            match key:

                case 'TITLE':
                    title = value[0]

                case 'POINTERS':
                    meta = self._read_pointers(raw_data[key])

                case 'ATOM_NAME':
                    atoms['name'] = self._read_atom_name(value)

                case 'CHARGE':
                    atoms['charge'] = self._read_charge(value)

                case 'ATOMIC_NUMBER':
                    atoms['element'] = self._read_atom_number(value)

                case 'MASS':
                    atoms['mass'] = self._read_mass(value)

                case 'ATOM_TYPE_INDEX':
                    atoms['type'] = self._read_atom_type(value)

                # case 'NUMBER_EXCLUDED_ATOMS':
                case 'NONBONDED_PARM_INDEX':
                    nonbonded_index, nonbonded_params, acoeff, bcoeff = self.parse_nonbonded_params(raw_data)

                # case 'RESIDUE_LABEL':
                case 'BOND_FORCE_CONSTANT':
                    value = ' '.join(value).split()
                    bond_params['force_constant'] = list(map(float, value))

                case 'BOND_EQUIL_VALUE':
                    value = ' '.join(value).split()
                    bond_params['equil_value'] = list(map(float, value))

                case 'ANGLE_FORCE_CONSTANT':
                    value = ' '.join(value).split()
                    angle_params['force_constant'] = list(map(float, value))
                
                case 'ANGLE_EQUIL_VALUE':
                    value = ' '.join(value).split()
                    angle_params['equil_value'] = list(map(float, value))

                case 'DIHEDRAL_FORCE_CONSTANT':
                    value = ' '.join(value).split()
                    dihedral_params['force_constant'] = list(map(float, value))

                case 'DIHEDRAL_PERIODICITY':
                    value = ' '.join(value).split()
                    dihedral_params['periodicity'] = list(map(float, value))

                case 'DIHEDRAL_PHASE':
                    value = ' '.join(value).split()
                    dihedral_params['phase'] = list(map(float, value))

                case 'LENNARD_JONES_ACOEF' | 'LENNARD_JONES_BCOEF':
                    pairs[key] = list(map(float, ' '.join(value).split()))

                case 'BONDS_INC_HYDROGEN' | 'BONDS_WITHOUT_HYDROGEN':
                    bond_idx = ' '.join(value).split()
                    bonds['i'].extend(idx for idx in bond_idx[0::3] )
                    bonds['j'].extend(idx for idx in bond_idx[1::3] )
                    bonds['type'].extend(idx for idx in bond_idx[2::3])

                case 'ANGLES_INC_HYDROGEN' | 'ANGLES_WITHOUT_HYDROGEN':
                    angle_idx = ' '.join(value).split()
                    angles['i'].extend(idx for idx in angle_idx[0::4] )
                    angles['j'].extend(idx for idx in angle_idx[1::4] )
                    angles['k'].extend(idx for idx in angle_idx[2::4] )
                    angles['type'].extend(idx for idx in angle_idx[3::4])

                case 'DIHEDRALS_INC_HYDROGEN' | 'DIHEDRALS_WITHOUT_HYDROGEN':
                    dihe_idx = ' '.join(value).split()
                    dihedrals['i'].extend(idx for idx in dihe_idx[0::5])
                    dihedrals['j'].extend(idx for idx in dihe_idx[1::5])
                    dihedrals['k'].extend(idx for idx in dihe_idx[2::5])
                    dihedrals['l'].extend(idx for idx in dihe_idx[3::5])
                    dihedrals['type'].extend(idx for idx in dihe_idx[4::5])

                # case 'SCEE_SCALE_FACTOR':
                # case 'SCNB_SCALE_FACTOR':
        atoms['id'] = np.arange(1, meta['n_atoms'] + 1, dtype=int)

        bonds['id'] = np.arange(1, meta['n_bonds'] + 1, dtype=int)
        bonds['type'] = np.array(bonds['type'], dtype=int)
        bonds['i'] = np.array(np.abs(np.array(bonds['i'], dtype=int)) / 3 + 1, dtype=int)
        bonds['j'] = np.array(np.abs(np.array(bonds['j'], dtype=int)) / 3 + 1, dtype=int)
        bond_style = system.forcefield.def_bondstyle("harmonic")

        unique_type, unique_idx = np.unique(bonds['type'], return_index=True)
        for idx, type_id in zip(unique_idx, unique_type):
            i = bonds['i'][idx] - 1
            j = bonds['j'][idx] - 1
            bond_style.def_type(type_id, atoms["type"][i], atoms["type"][j], force_constant=bond_params['force_constant'][type_id-1], equil_value=bond_params['equil_value'][type_id-1])

        angles['id'] = np.arange(1, meta['n_angles'] + 1, dtype=int)
        angles['type'] = np.array(angles['type'], dtype=int)
        angles['i'] = np.array(np.abs(np.array(angles['i'], dtype=int)) / 3 + 1, dtype=int)
        angles['j'] = np.array(np.abs(np.array(angles['j'], dtype=int)) / 3 + 1, dtype=int)
        angles['k'] = np.array(np.abs(np.array(angles['k'], dtype=int)) / 3 + 1, dtype=int)
        angle_style = system.forcefield.def_anglestyle("harmonic")

        unique_type, unique_idx = np.unique(angles['type'], return_index=True)
        for idx, type_id in zip(unique_idx, unique_type):
            i = angles['i'][idx] - 1
            j = angles['j'][idx] - 1
            k = angles['k'][idx] - 1
            angle_style.def_type(type_id, atoms["type"][i], atoms["type"][j], atoms["type"][k], force_constant=angle_params['force_constant'][type_id-1], equil_value=angle_params['equil_value'][type_id-1])
        
        dihedrals['id'] = np.arange(1, meta['n_dihedrals'] + 1, dtype=int)
        dihedrals['type'] = np.array(dihedrals['type'], dtype=int)

        mhar = system.forcefield.def_dihedralstyle("multi/harmonic")
        charmm = system.forcefield.def_dihedralstyle("charmm")
        for i, j, k, l, type_id in zip(dihedrals['i'], dihedrals['j'], dihedrals['k'], dihedrals['l'], dihedrals['type']):

            if l < 0:
                # if the fourth atom is negative, this implies that the dihedral is an improper.
                pass
            if k < 0:
                # If the third atom is negative, this implies that the end group interations are to be ignored.
                pass


            if dihedral_style.get_type(type_id):
                continue
            dihedral_style.def_type(type_id, atoms["type"][i-1], atoms["type"][j-1], atoms["type"][k-1], atoms["type"][l-1], force_constant=dihedral_params['force_constant'][type_id-1], periodicity=dihedral_params['periodicity'][type_id-1], phase=dihedral_params['phase'][type_id-1])


        dihedrals['i'] = np.array(np.abs(np.array(dihedrals['i'], dtype=int)) / 3 + 1, dtype=int)
        dihedrals['j'] = np.array(np.abs(np.array(dihedrals['j'], dtype=int)) / 3 + 1, dtype=int)
        dihedrals['k'] = np.array(np.abs(np.array(dihedrals['k'], dtype=int)) / 3 + 1, dtype=int)
        dihedrals['l'] = np.array(np.abs(np.array(dihedrals['l'], dtype=int)) / 3 + 1, dtype=int)




        atoms, bonds, angles, dihedrals = self._parse_residues(raw_data['RESIDUE_POINTER'], meta, atoms, bonds, angles, dihedrals)

        system.frame['props'] = meta
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
        meta_data['n_atoms'] = meta_data['NATOM']
        meta_data['n_bonds'] = meta_data['NBONH'] + meta_data['MBONA']
        meta_data['n_angles'] = meta_data['NTHETH'] + meta_data['MTHETA']
        meta_data['n_dihedrals'] = meta_data['NPHIH'] + meta_data['MPHIA']
        meta_data['n_atomtypes'] = meta_data['NATYP']
        meta_data['n_bondtypes'] = meta_data['NUMBND']
        meta_data['n_angletypes'] = meta_data['NUMANG']
        meta_data['n_dihedraltypes'] = meta_data['NPTRA']
        return meta_data
    
    def _read_atom_name(self, lines: list[str]):
        
        names = []

        for line in lines:
            names.extend(line[i: i+4].strip() for i in range(0, len(line), 4))

        return names

    def _read_charge(self, lines: list[str]):

        charges = []
        for line in lines:
            charges.extend(map(float, line.split()))

        return charges

    def _read_atom_number(self, lines: list[str]):

        atom_numbers = []
        for line in lines:
            atom_numbers.extend(map(int, line.split()))

        return atom_numbers

    def _read_mass(self, lines: list[str]):

        masses = []
        for line in lines:
            masses.extend(map(float, line.split()))

        return masses
    
    def _read_atom_type(self, lines: list[str]):

        atom_types = []
        for line in lines:
            atom_types.extend(map(int, line.split()))

        return atom_types
    
    def _parse_residues(self, pointer, meta, atoms, bonds, angles, dihedrals):

        residue_slice = list(accumulate(map(lambda x: int(x)-1, pointer))) + [meta['n_atoms']]  # pointer is 1-indexed
        n_atoms = meta['n_atoms']
        assert n_atoms == residue_slice[-1], f'Number of atoms does not match residue pointers, {n_atoms} != {residue_slice[-1]}'
        segment_lengths = np.diff(residue_slice)
        atom_residue_mask = np.repeat(np.arange(len(pointer)), segment_lengths)

        # get bond mask: if both i and j in atom_mask, then bond is intra-residue and equal to atom mask in corresponding index, else inter-residue and -1
        bond_i = bonds['i']
        bond_j = bonds['j']
        bond_residue_mask = np.where(np.isin(bond_i, atom_residue_mask) & np.isin(bond_j, atom_residue_mask), atom_residue_mask[bond_i], -1)

        atoms['residue'] = atom_residue_mask
        bonds['residue'] = bond_residue_mask

        angle_i = angles['i']
        angle_j = angles['j']
        angle_k = angles['k']
        angle_residue_mask = np.where(np.isin(angle_i, atom_residue_mask) & np.isin(angle_j, atom_residue_mask) & np.isin(angle_k, atom_residue_mask), atom_residue_mask[angle_i], -1)

        angles['residue'] = angle_residue_mask

        return atoms, bonds, angles, dihedrals


    def parse_nonbonded_params(self, raw_data: dict):
        nonbonded_params = {}
        nonbonded_index = list(map(int, (' '.join(raw_data['NONBONDED_PARM_INDEX'])).split()))
        acoef = list(map(float, (' '.join(raw_data['LENNARD_JONES_ACOEF'])).split()))
        bcoef = list(map(float, (' '.join(raw_data['LENNARD_JONES_BCOEF'])).split()))
        
        for i in range(len(nonbonded_index)):
            index = nonbonded_index[i] - 1 
            if index >= 0:
                a_param = acoef[index]
                b_param = bcoef[index]
                nonbonded_params[i + 1] = (a_param, b_param)

        return nonbonded_index, nonbonded_params, acoef, bcoef