from pathlib import Path
import molpy as mp
import numpy as np
from typing import Iterator
import pyarrow as pa
from itertools import accumulate
from math import sqrt

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
    
    @staticmethod
    def read_section(lines: Iterator[str], convert_fn: int) -> list[str]:

        return list(map(convert_fn, ' '.join(lines).split()))

    def read(self, system: mp.System) -> mp.ForceField:

        with open(self.file, "r") as f:
            
            lines = filter(lambda line: line, map(AmberPrmtopReader.sanitizer, f.readlines()))

        # read file and split into sections
        self.raw_data = {}
        data = []
        flag = None

        for line in lines:

            if line.startswith(f'%FLAG'):
                if flag:
                    if flag in self.raw_data:
                        self.raw_data[flag].extend(data)
                    else:
                        self.raw_data[flag] = data

                flag = line.split()[1]
                data = []

            elif line.startswith(f'%FORMAT'):
                pass

            else:
                data.append(line)

        if flag:
            self.raw_data[flag] = data

        # parse string and get data
        self.meta = {}
        atoms = {}
        bonds = {
            'type': [],
            'type_name': [],
            'i': [],
            'j': [],
        }
        angles = {
            'type': [],
            'type_name': [],
            'i': [],
            'j': [],
            'k': [],
        }
        dihedrals = {
            'type': [],
            'type_name': [],
            'i': [],
            'j': [],
            'k': [],
            'l': [],
        }
        pairs = {}

        for key, value in self.raw_data.items():
            match key:

                case 'TITLE':
                    self.meta['title'] = value[0]

                case 'POINTERS':
                    self.meta = meta = self._read_pointers(self.raw_data[key])

                case 'ATOM_NAME':
                    atoms['name'] = self._read_atom_name(value)

                case 'CHARGE':
                    atoms['charge'] = self.read_section(value, float)

                case 'ATOMIC_NUMBER':
                    atoms['element'] = self.read_section(value, int)

                case 'MASS':
                    atoms['mass'] = self.read_section(value, float)

                case 'ATOM_TYPE_INDEX':
                    atoms['type'] = self.read_section(value, int)

                case 'AMBER_ATOM_TYPE':
                    atoms['type_name'] = self.read_section(value, str)

                case 'BOND_FORCE_CONSTANT' | 'BOND_EQUIL_VALUE' | 'ANGLE_FORCE_CONSTANT' | 'ANGLE_EQUIL_VALUE' | 'DIHEDRAL_FORCE_CONSTANT' | 'DIHEDRAL_PERIODICITY' | 'DIHEDRAL_PHASE' | 'LENNARD_JONES_ACOEF' | 'LENNARD_JONES_BCOEF':

                    self.raw_data[key] = self.read_section(value, float)

                case 'BONDS_INC_HYDROGEN' | 'BONDS_WITHOUT_HYDROGEN' | 'ANGLES_INC_HYDROGEN' | 'ANGLES_WITHOUT_HYDROGEN' | 'DIHEDRALS_INC_HYDROGEN' | 'DIHEDRALS_WITHOUT_HYDROGEN' | 'NONBONDED_PARM_INDEX':
                    self.raw_data[key] = self.read_section(value, int)

        # def forcefield
        atoms['id'] = np.arange(meta['n_atoms'], dtype=int) + 1
        atoms['charge'] = np.array(atoms['charge']) / 18.2223
        system.forcefield.units = 'real'
        atomstyle = system.forcefield.def_atomstyle('full')
        atomtypes = atomstyle.types
        for itype, name in zip(atoms['type'], atoms['type_name']):
            atomstyle.def_type(name, kw_params={"id": itype})

        bondstyle = system.forcefield.def_bondstyle("harmonic")
        for bond_type, i, j, f, r_min in self.get_bond_with_H() + self.get_bond_without_H():
            atom_i_type_name = atoms['type_name'][i-1]
            atom_j_type_name = atoms['type_name'][j-1]
            bond_name = f'{atom_i_type_name}-{atom_j_type_name}'
            if bond_name not in bondstyle.types:
                bondstyle.def_type(bond_name, atomtypes[atom_i_type_name], atomtypes[atom_j_type_name], kw_params={'force_constant':f, 'equil_value':r_min, 'id': bond_type})  # if multiply by 2
            bonds['type'].append(bond_type)
            bonds['i'].append(i)
            bonds['j'].append(j)
            bonds['type_name'].append(bond_name)
        bonds['id'] = np.arange(meta['n_bonds'], dtype=int) + 1

        anglestyle = system.forcefield.def_anglestyle("harmonic")
        for angle_type, i, j, k, f, theta_min in self.parse_angle_params():
            atom_i_type_name = atoms['type_name'][i-1]
            atom_j_type_name = atoms['type_name'][j-1]
            atom_k_type_name = atoms['type_name'][k-1]
            angle_name = f'{atom_i_type_name}-{atom_j_type_name}-{atom_k_type_name}'
            if angle_name not in anglestyle.types:
                anglestyle.def_type(angle_name, atomtypes[atom_i_type_name], atomtypes[atom_j_type_name], atomtypes[atom_k_type_name], kw_params={'force_constant':f, 'equil_value':theta_min, 'id': angle_type})
            
            angles['type'].append(angle_type)
            angles['i'].append(i)
            angles['j'].append(j)
            angles['k'].append(k)
            angles['type_name'].append(angle_name)

        angles['id'] = np.arange(meta['n_angles'], dtype=int) + 1

        dihedralstyle = system.forcefield.def_dihedralstyle("charmmfsw")
        for dihe_type, i, j, k, l, f, phase, periodicity in self.parse_dihedral_params():
            atom_i_type_name = atoms['type_name'][i-1]
            atom_j_type_name = atoms['type_name'][j-1]
            atom_k_type_name = atoms['type_name'][k-1]
            atom_l_type_name = atoms['type_name'][l-1]
            dihe_name = f'{atom_i_type_name}-{atom_j_type_name}-{atom_k_type_name}-{atom_l_type_name}'
            if dihe_name not in dihedralstyle.types:
                dihedralstyle.def_type(dihe_name, atomtypes[atom_i_type_name], atomtypes[atom_j_type_name], atom_k_type_name, atomtypes[atom_l_type_name], kw_params={'force_constant':f, 'phase':phase, 'periodicity':periodicity, 'id': dihe_type})
            dihedrals['type'].append(dihe_type)
            dihedrals['i'].append(i)
            dihedrals['j'].append(j)
            dihedrals['k'].append(k)
            dihedrals['l'].append(l)
            dihedrals['type_name'].append(dihe_name)
        dihedrals['id'] = np.arange(meta['n_dihedrals'], dtype=int) + 1

        atoms, bonds, angles, dihedrals = self._parse_residues(self.raw_data['RESIDUE_POINTER'], meta, atoms, bonds, angles, dihedrals)

        pairstyle = system.forcefield.def_pairstyle("lj/cut/coul/long", order_params=[9.0, 9.0])
        for itype, rVdw, epsilon in self.parse_nonbond_params(atoms):
            atom_i_type_name = atoms['type_name'][itype-1]
            atom_j_type_name = atoms['type_name'][itype-1]
            pair_name = f'{atom_i_type_name}-{atom_j_type_name}'
            pairstyle.def_type(pair_name, atom_i_type_name, atom_j_type_name, kw_params={'rVdw': rVdw, 'epsilon': epsilon, 'id': itype})

        # store in system
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
    
    def _parse_residues(self, pointer, meta, atoms, bonds, angles, dihedrals):

        residue_slice = list(accumulate(map(lambda x: int(x)-1, pointer))) + [meta['n_atoms']]  # pointer is 1-indexed
        n_atoms = meta['n_atoms']
        assert n_atoms == residue_slice[-1], f'Number of atoms does not match residue pointers, {n_atoms} != {residue_slice[-1]}'
        segment_lengths = np.diff(residue_slice)
        atom_residue_mask = np.repeat(np.arange(len(pointer))+1, segment_lengths)

        # get bond mask: if both i and j in atom_mask, then bond is intra-residue and equal to atom mask in corresponding index, else inter-residue and -1
        bond_i = bonds['i']
        bond_j = bonds['j']
        bond_residue_mask = np.zeros(len(bond_i), dtype=int)
        for residue in np.unique(atom_residue_mask):
            atom_mask = atom_residue_mask == residue  # atoms' id in residue
            bond_residue_mask[np.where(np.isin(bond_i, atom_mask) & np.isin(bond_j, atom_mask))] = residue


        atoms['residue'] = atom_residue_mask
        bonds['residue'] = bond_residue_mask

        angle_i = angles['i']
        angle_j = angles['j']
        angle_k = angles['k']
        angle_residue_mask = np.where(np.isin(angle_i, atom_residue_mask) & np.isin(angle_j, atom_residue_mask) & np.isin(angle_k, atom_residue_mask), atom_residue_mask[angle_i], -1)

        angles['residue'] = angle_residue_mask

        return atoms, bonds, angles, dihedrals
    
    def _parse_bond_params(self, bondPointers):
        forceConstant=self.raw_data["BOND_FORCE_CONSTANT"]
        bondEquil=self.raw_data["BOND_EQUIL_VALUE"]
        returnList=[]
        # forceConstConversionFactor = (units.kilocalorie_per_mole/(units.angstrom*units.angstrom)).conversion_factor_to(units.kilojoule_per_mole/(units.nanometer*units.nanometer))
        # lengthConversionFactor = units.angstrom.conversion_factor_to(units.nanometer)
        for ii in range(0,len(bondPointers),3):
            if int(bondPointers[ii])<0 or \
            int(bondPointers[ii+1])<0:
                raise Exception("Found negative bonded atom pointers %s"
                                % ((bondPointers[ii],
                                    bondPointers[ii+1]),))
            iType=int(bondPointers[ii+2])
            i, j = sorted((int(bondPointers[ii])//3+1, int(bondPointers[ii+1])//3+1))

            returnList.append((iType,  # return 1-based idx
                            i,
                            j,
                            float(forceConstant[iType-1]),
                            float(bondEquil[iType-1])))
        return returnList


    def get_bond_with_H(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with a hydrogen"""
        bondPointers=self.raw_data["BONDS_INC_HYDROGEN"]
        return self._parse_bond_params(bondPointers)


    def get_bond_without_H(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with no hydrogen"""
        bondPointers=self.raw_data["BONDS_WITHOUT_HYDROGEN"]
        return self._parse_bond_params(bondPointers)

    
    def parse_angle_params(self):
        """Return list of atom triplets, K, and ThetaMin for each bond angle"""
        try:
            return self._angleList
        except AttributeError:
            pass
        forceConstant=self.raw_data["ANGLE_FORCE_CONSTANT"]
        angleEquil=self.raw_data["ANGLE_EQUIL_VALUE"]
        anglePointers = self.raw_data["ANGLES_INC_HYDROGEN"] \
                       +self.raw_data["ANGLES_WITHOUT_HYDROGEN"]
        self._angleList=[]
        # forceConstConversionFactor = (units.kilocalorie_per_mole/(units.radian*units.radian)).conversion_factor_to(units.kilojoule_per_mole/(units.radian*units.radian))
        for ii in range(0,len(anglePointers),4):
             if int(anglePointers[ii])<0 or \
                int(anglePointers[ii+1])<0 or \
                int(anglePointers[ii+2])<0:
                 raise Exception("Found negative angle atom pointers %s"
                                 % ((anglePointers[ii],
                                     anglePointers[ii+1],
                                     anglePointers[ii+2]),))
             iType=int(anglePointers[ii+3])
             i, k = sorted((int(anglePointers[ii])//3+1, int(anglePointers[ii+2])//3+1))
             self._angleList.append((iType, i,
                                int(anglePointers[ii+1])//3+1,
                                k,
                                float(forceConstant[iType-1]),
                                float(angleEquil[iType-1])))
        return self._angleList


    def parse_dihedral_params(self):
        """Return list of atom quads, K, phase and periodicity for each dihedral angle"""

        forceConstant=self.raw_data["DIHEDRAL_FORCE_CONSTANT"]
        phase=self.raw_data["DIHEDRAL_PHASE"]
        periodicity=self.raw_data["DIHEDRAL_PERIODICITY"]
        dihedralPointers = self.raw_data["DIHEDRALS_INC_HYDROGEN"] \
                          +self.raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
        self._dihedralList=[]
        # forceConstConversionFactor = (units.kilocalorie_per_mole).conversion_factor_to(units.kilojoule_per_mole)
        for ii in range(0,len(dihedralPointers),5):
             if int(dihedralPointers[ii])<0 or int(dihedralPointers[ii+1])<0:
                 raise Exception("Found negative dihedral atom pointers %s"
                                 % ((dihedralPointers[ii],
                                    dihedralPointers[ii+1],
                                    dihedralPointers[ii+2],
                                    dihedralPointers[ii+3]),))
             iType=int(dihedralPointers[ii+4])
             i, j, k, l = int(dihedralPointers[ii])//3+1, \
                                int(dihedralPointers[ii+1])//3+1, \
                                abs(int(dihedralPointers[ii+2]))//3+1, \
                                abs(int(dihedralPointers[ii+3]))//3+1
             if j > k:
                i, j, k, l = l, k, j, i
             self._dihedralList.append((iType, i, j, k, l,
                                float(forceConstant[iType-1]),
                                float(phase[iType-1]),
                                int(0.5+float(periodicity[iType-1]))))
        return self._dihedralList

    # def getImpropers(self):
    #     """Return list of atom quads, K, and phase for each improper torsion"""
    #     try:
    #         return self._improperList
    #     except AttributeError:
    #         pass
    #     self._improperList = []
    #     if 'CHARMM_IMPROPERS' in self.raw_data:
    #         forceConstant = self.raw_data["CHARMM_IMPROPER_FORCE_CONSTANT"]
    #         phase = self.raw_data["CHARMM_IMPROPER_PHASE"]
    #         improperPointers = self.raw_data["CHARMM_IMPROPERS"]
    #         # forceConstConversionFactor = (units.kilocalorie_per_mole).conversion_factor_to(units.kilojoule_per_mole)
    #         for ii in range(0,len(improperPointers),5):
    #              if int(improperPointers[ii])<0 or int(improperPointers[ii+1])<0:
    #                  raise Exception("Found negative improper atom pointers %s"
    #                                  % ((improperPointers[ii],
    #                                     improperPointers[ii+1],
    #                                     improperPointers[ii+2],
    #                                     improperPointers[ii+3]),))
    #              iType = int(improperPointers[ii+4])-1
    #              self._improperList.append((int(improperPointers[ii])-1,
    #                                 int(improperPointers[ii+1])-1,
    #                                 abs(int(improperPointers[ii+2]))-1,
    #                                 abs(int(improperPointers[ii+3]))-1,
    #                                 float(forceConstant[iType]),# *forceConstConversionFactor,
    #                                 float(phase[iType])))
    #     return self._improperList

    def parse_nonbond_params(self,atoms):
        """
        Return list of all rVdw, epsilon pairs for each atom. If off-diagonal
        elements of the Lennard-Jones A and B coefficient matrices are found,
        NbfixPresent exception is raised
        """
        # if self._has_nbfix_terms:
        #     raise Exception('Off-diagonal Lennard-Jones elements found. '
        #                 'Cannot determine LJ parameters for individual atoms.')

        # Check if there are any non-zero HBOND terms
        for x, y in zip(self.raw_data['HBOND_ACOEF'], self.raw_data['HBOND_BCOEF']):
            if float(x) or float(y):
                raise Exception('10-12 interactions are not supported')
        return_list=[]
        # lengthConversionFactor = units.angstrom.conversion_factor_to(units.nanometer)
        # energyConversionFactor = units.kilocalorie_per_mole.conversion_factor_to(units.kilojoule_per_mole)
        numTypes = self.meta['NTYPES']
        atomTypeIndexes=atoms['type']
        type_parameters = [(0, 0) for i in range(numTypes)]
        for iAtom in range(self.meta['NATOM']):
            index=(numTypes+1)*(atomTypeIndexes[iAtom]-1)
            nbIndex=int(self.raw_data['NONBONDED_PARM_INDEX'][index])-1
            if nbIndex<0:
                raise Exception("10-12 interactions are not supported")
            acoef = float(self.raw_data['LENNARD_JONES_ACOEF'][nbIndex])
            bcoef = float(self.raw_data['LENNARD_JONES_BCOEF'][nbIndex])
            try:
                rMin = (2*acoef/bcoef)**(1/6.0)
                epsilon = 0.25*bcoef*bcoef/acoef
            except ZeroDivisionError:
                rMin = 1.0
                epsilon = 0.0
            type_parameters[atomTypeIndexes[iAtom]-1] = (rMin/2.0, epsilon)
            # jichen: unit conversion 
            # length: angstrom to namometer
            # epsilon: kcal/mol to kJ/mol
            rVdw = rMin/2.0
            return_list.append( (iAtom+1, rVdw, epsilon) )
        # Check if we have any off-diagonal modified LJ terms that would require
        # an NBFIX-like solution
        # for i in range(numTypes):
        #     for j in range(numTypes):
        #         index = int(self.raw_data['NONBONDED_PARM_INDEX'][numTypes*i+j]) - 1
        #         if index < 0: continue
        #         rij = type_parameters[i][0] + type_parameters[j][0]
        #         wdij = sqrt(type_parameters[i][1] * type_parameters[j][1])
        #         a = float(self.raw_data['LENNARD_JONES_ACOEF'][index])
        #         b = float(self.raw_data['LENNARD_JONES_BCOEF'][index])
        #         if a == 0 or b == 0:
        #             if a != 0 or b != 0 or (wdij != 0 and rij != 0):
        #                 self._has_nbfix_terms = True
        #                 raise Exception('Off-diagonal Lennard-Jones elements'
        #                                    ' found. Cannot determine LJ '
        #                                    'parameters for individual atoms.')
        #         elif (abs((a - (wdij * rij ** 12)) / a) > 1e-6 or
        #               abs((b - (2 * wdij * rij**6)) / b) > 1e-6):
        #             self._has_nbfix_terms = True
        #             raise Exception('Off-diagonal Lennard-Jones elements '
        #                                'found. Cannot determine LJ parameters '
        #                                'for individual atoms.')
        return return_list