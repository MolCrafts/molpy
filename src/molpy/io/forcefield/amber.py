import math
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np

import molpy as mp


class AmberPrmtopReader:
    def __init__(
        self,
        file: str | Path,
    ):
        self.file = file

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.strip()

    @staticmethod
    def read_section(lines: Iterator[str], convert_fn: Callable = int) -> list[str]:
        return list(map(convert_fn, " ".join(lines).split()))

    def read(self, frame: mp.Frame):
        with open(self.file) as f:
            lines = filter(
                lambda line: line, map(AmberPrmtopReader.sanitizer, f.readlines())
            )

        # read file and split into sections
        self.raw_data = {}
        data = []
        flag = None

        for line in lines:
            if line.startswith("%FLAG"):
                if flag:
                    if flag in self.raw_data:
                        self.raw_data[flag].extend(data)
                    else:
                        self.raw_data[flag] = data

                flag = line.split()[1]
                data = []

            elif line.startswith("%FORMAT"):
                pass

            else:
                data.append(line)

        if flag:
            self.raw_data[flag] = data

        # parse string and get data
        self.meta = {}
        atoms = {}
        bonds = {
            "type_id": [],
            "type": [],
            "i": [],
            "j": [],
        }
        angles = {
            "type_id": [],
            "type": [],
            "i": [],
            "j": [],
            "k": [],
        }
        dihedrals = {
            "type_id": [],
            "type": [],
            "i": [],
            "j": [],
            "k": [],
            "l": [],
        }

        for key, value in self.raw_data.items():
            match key:
                case "TITLE":
                    self.meta["title"] = value[0]

                case "POINTERS":
                    self.meta = meta = self._read_pointers(self.raw_data[key])

                case "ATOM_NAME":
                    atoms["name"] = self._read_atom_name(value)

                case "CHARGE":
                    atoms["q"] = self.read_section(value, float)

                case "ATOMIC_NUMBER":
                    atoms["atomic_number"] = self.read_section(value, int)

                case "MASS":
                    atoms["mass"] = self.read_section(value, float)

                case "ATOM_TYPE_INDEX":
                    self.raw_data[key] = self.read_section(value, int)

                case "AMBER_ATOM_TYPE":
                    atoms["type"] = self.read_section(value, str)

                case (
                    "BOND_FORCE_CONSTANT"
                    | "BOND_EQUIL_VALUE"
                    | "ANGLE_FORCE_CONSTANT"
                    | "ANGLE_EQUIL_VALUE"
                    | "DIHEDRAL_FORCE_CONSTANT"
                    | "DIHEDRAL_PERIODICITY"
                    | "DIHEDRAL_PHASE"
                    | "LENNARD_JONES_ACOEF"
                    | "LENNARD_JONES_BCOEF"
                ):
                    self.raw_data[key] = self.read_section(value, float)

                case (
                    "BONDS_INC_HYDROGEN"
                    | "BONDS_WITHOUT_HYDROGEN"
                    | "ANGLES_INC_HYDROGEN"
                    | "ANGLES_WITHOUT_HYDROGEN"
                    | "DIHEDRALS_INC_HYDROGEN"
                    | "DIHEDRALS_WITHOUT_HYDROGEN"
                    | "NONBONDED_PARM_INDEX"
                ):
                    self.raw_data[key] = self.read_section(value, int)

        # def forcefield
        atoms["id"] = np.arange(meta["n_atoms"], dtype=int) + 1
        atoms["q"] = np.array(atoms["q"]) / 18.2223
        ff = mp.ForceField()
        ff.units = "real"
        atomstyle = ff.def_atomstyle("full")
        atomtype_map = {}  # atomtype id : atomtype
        for itype, name, mass in zip(
            self.raw_data["ATOM_TYPE_INDEX"], atoms["type"], atoms["mass"]
        ):
            # amber atom type and type index not 1 to 1 mapping involving different files
            if name not in atomtype_map:
                atomtype_map[name] = atomstyle.def_type(name, id=itype, mass=mass)

        bondstyle = ff.def_bondstyle("harmonic")
        for bond_type, i, j, f, r_min in (
            self.get_bond_with_H() + self.get_bond_without_H()
        ):
            atom_i_type_name = atoms["type"][i - 1]
            atom_j_type_name = atoms["type"][j - 1]
            bond_name = "-".join(sorted([atom_i_type_name, atom_j_type_name]))
            if bond_name not in bondstyle.types:
                bondstyle.def_type(
                    atomtype_map[atom_i_type_name],
                    atomtype_map[atom_j_type_name],
                    name=bond_name,
                    force_constant=f,
                    equil_value=r_min,
                    id=bond_type,
                    parms=[f, r_min],
                )  # if multiply by 2
            bonds["type_id"].append(bond_type)
            bonds["i"].append(i - 1)
            bonds["j"].append(j - 1)
            bonds["type"].append(bond_name)
        bonds["id"] = np.arange(meta["n_bonds"], dtype=int) + 1

        anglestyle = ff.def_anglestyle("harmonic")
        for angle_type, i, j, k, f, theta_min in self.parse_angle_params():
            atom_i_type_name = atoms["type"][i - 1]
            atom_j_type_name = atoms["type"][j - 1]
            atom_k_type_name = atoms["type"][k - 1]
            atom_i_type_name, atom_k_type_name = sorted(
                [atom_i_type_name, atom_k_type_name]
            )
            angle_name = f"{atom_i_type_name}-{atom_j_type_name}-{atom_k_type_name}"
            if angle_name not in anglestyle.types:
                anglestyle.def_type(
                    atomtype_map[atom_i_type_name],
                    atomtype_map[atom_j_type_name],
                    atomtype_map[atom_k_type_name],
                    name=angle_name,
                    force_constant=f,
                    equil_value=theta_min,
                    id=angle_type,
                    parms=[f, theta_min],
                )

            angles["type_id"].append(angle_type)
            angles["i"].append(i - 1)
            angles["j"].append(j - 1)
            angles["k"].append(k - 1)
            angles["type"].append(angle_name)

        angles["id"] = np.arange(meta["n_angles"], dtype=int) + 1

        dihedralstyle = ff.def_dihedralstyle("charmmfsw")
        for (
            dihe_type,
            i,
            j,
            k,
            l,
            f,
            phase,
            periodicity,
        ) in self.parse_dihedral_params():
            atom_i_type_name = atoms["type"][i - 1]
            atom_j_type_name = atoms["type"][j - 1]
            atom_k_type_name = atoms["type"][k - 1]
            atom_l_type_name = atoms["type"][l - 1]
            if atom_j_type_name > atom_k_type_name:
                atom_j_type_name, atom_k_type_name = atom_k_type_name, atom_j_type_name
                atom_i_type_name, atom_l_type_name = atom_l_type_name, atom_i_type_name
            dihe_name = f"{atom_i_type_name}-{atom_j_type_name}-{atom_k_type_name}-{atom_l_type_name}"
            if dihe_name not in dihedralstyle.types:
                dihedralstyle.def_type(
                    atomtype_map[atom_i_type_name],
                    atomtype_map[atom_j_type_name],
                    atomtype_map[atom_k_type_name],
                    atomtype_map[atom_l_type_name],
                    name=dihe_name,
                    force_constant=f,
                    phase=phase,
                    periodicity=periodicity,
                    id=dihe_type,
                    parms=[f, periodicity, int(phase), 0.5],
                )
            dihedrals["type_id"].append(dihe_type)
            dihedrals["i"].append(i - 1)
            dihedrals["j"].append(j - 1)
            dihedrals["k"].append(k - 1)
            dihedrals["l"].append(l - 1)
            dihedrals["type"].append(dihe_name)
        dihedrals["id"] = np.arange(meta["n_dihedrals"], dtype=int) + 1

        atoms, bonds, angles, dihedrals = self._parse_residues(
            self.raw_data["RESIDUE_POINTER"], meta, atoms, bonds, angles, dihedrals
        )

        pairstyle = ff.def_pairstyle("lj/charmmfsw/coul/charmmfsh", [2.5, 9.0])
        for itype, rVdw, epsilon in self.parse_nonbond_params(atoms):
            atom_i_type_name = atoms["type"][itype - 1]
            atom_j_type_name = atoms["type"][itype - 1]
            pair_name = f"{atom_i_type_name}-{atom_j_type_name}"
            pairstyle.def_type(
                atomtype_map[atom_i_type_name],
                atomtype_map[atom_j_type_name],
                name=pair_name,
                parms=[epsilon, rVdw],
                rVdw=rVdw,
                epsilon=epsilon,
                id=itype,
            )

        # store in frame
        frame.metadata.update(meta)
        frame["atoms"] = atoms
        frame["bonds"] = bonds
        frame["angles"] = angles
        frame["dihedrals"] = dihedrals

        return frame, ff

    def _read_pointers(self, lines):
        meta_fields = (
            "NATOM",
            "NTYPES",
            "NBONH",
            "MBONA",
            "NTHETH",
            "MTHETA",
            "NPHIH",
            "MPHIA",
            "NHPARM",
            "NPARM",
            "NNB",
            "NRES",
            "NBONA",
            "NTHETA",
            "NPHIA",
            "NUMBND",
            "NUMANG",
            "NPTRA",
            "NATYP",
            "NPHB",
            "IFPERT",
            "NBPER",
            "NGPER",
            "NDPER",
            "MBPER",
            "MGPER",
            "MDPER",
            "IFBOX",
            "NMXRS",
            "IFCAP",
            "NUMEXTRA",
            "NCOPY",
        )
        meta_data = dict(zip(meta_fields, map(int, " ".join(lines).split())))
        meta = {}
        meta["n_atoms"] = meta_data["NATOM"]
        meta["n_bonds"] = meta_data["NBONH"] + meta_data["MBONA"]
        meta["n_angles"] = meta_data["NTHETH"] + meta_data["MTHETA"]
        meta["n_dihedrals"] = meta_data["NPHIH"] + meta_data["MPHIA"]
        meta["n_atomtypes"] = meta_data["NATYP"]
        meta["n_bondtypes"] = meta_data["NUMBND"]
        meta["n_angletypes"] = meta_data["NUMANG"]
        meta["n_dihedraltypes"] = meta_data["NPTRA"]
        meta.update(meta_data)
        return meta

    def _read_atom_name(self, lines: list[str]):
        names = []

        for line in lines:
            names.extend(line[i : i + 4].strip() for i in range(0, len(line), 4))

        return names

    def _parse_residues(self, pointer, meta, atoms, bonds, angles, dihedrals):
        pointer = " ".join(pointer).split()
        pointer.append(meta["n_atoms"] + 1)
        # residue_slice = list((map(lambda x: int(x) - 1, pointer))) + [
        #     meta["n_atoms"]
        # ]  # pointer is 1-indexed
        # n_atoms = meta["n_atoms"]
        # assert (
        #     n_atoms == residue_slice[-1]
        # ), f"Number of atoms does not match residue pointers, {n_atoms} != {residue_slice[-1]}"
        # segment_lengths = np.diff(residue_slice)
        # atom_residue_mask = np.repeat(np.arange(len(pointer)) + 1, segment_lengths)
        residue_slice = np.array(pointer, dtype=int) - 1
        segment_lengths = np.diff(residue_slice)
        atom_residue_mask = np.repeat(
            np.arange(len(segment_lengths - 1)), segment_lengths
        )

        # get bond mask: if both i and j in atom_mask, then bond is intra-residue and equal to atom mask in corresponding index, else inter-residue and -1
        bond_i = bonds["i"]
        bond_j = bonds["j"]
        bond_residue_mask = np.zeros(len(bond_i), dtype=int)
        for residue in np.unique(atom_residue_mask):
            atom_mask = atom_residue_mask == residue  # atoms' id in residue
            bond_residue_mask[
                np.where(np.isin(bond_i, atom_mask) & np.isin(bond_j, atom_mask))
            ] = residue

        atoms["residue"] = atom_residue_mask
        bonds["residue"] = bond_residue_mask

        angle_i = angles["i"]
        angle_j = angles["j"]
        angle_k = angles["k"]
        angle_residue_mask = np.where(
            np.isin(angle_i, atom_residue_mask)
            & np.isin(angle_j, atom_residue_mask)
            & np.isin(angle_k, atom_residue_mask),
            atom_residue_mask[angle_i],
            -1,
        )

        angles["residue"] = angle_residue_mask

        return atoms, bonds, angles, dihedrals

    def _parse_bond_params(self, bondPointers):
        forceConstant = self.raw_data["BOND_FORCE_CONSTANT"]
        bondEquil = self.raw_data["BOND_EQUIL_VALUE"]
        returnList = []
        # forceConstConversionFactor = (units.kilocalorie_per_mole/(units.angstrom*units.angstrom)).conversion_factor_to(units.kilojoule_per_mole/(units.nanometer*units.nanometer))
        # lengthConversionFactor = units.angstrom.conversion_factor_to(units.nanometer)
        for ii in range(0, len(bondPointers), 3):
            if int(bondPointers[ii]) < 0 or int(bondPointers[ii + 1]) < 0:
                raise Exception(
                    f"Found negative bonded atom pointers {(bondPointers[ii], bondPointers[ii + 1])}"
                )
            iType = int(bondPointers[ii + 2])
            i, j = sorted(
                (int(bondPointers[ii]) // 3 + 1, int(bondPointers[ii + 1]) // 3 + 1)
            )

            returnList.append(
                (
                    iType,  # return 1-based idx
                    i,
                    j,
                    float(forceConstant[iType - 1]),
                    float(bondEquil[iType - 1]),
                )
            )
        return returnList

    def get_bond_with_H(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with a hydrogen"""
        bondPointers = self.raw_data["BONDS_INC_HYDROGEN"]
        return self._parse_bond_params(bondPointers)

    def get_bond_without_H(self):
        """Return list of bonded atom pairs, K, and Rmin for each bond with no hydrogen"""
        bondPointers = self.raw_data["BONDS_WITHOUT_HYDROGEN"]
        return self._parse_bond_params(bondPointers)

    def parse_angle_params(self):
        """Return list of atom triplets, K, and ThetaMin for each bond angle"""
        try:
            return self._angleList
        except AttributeError:
            pass
        forceConstant = self.raw_data["ANGLE_FORCE_CONSTANT"]
        angleEquil = self.raw_data["ANGLE_EQUIL_VALUE"]
        anglePointers = (
            self.raw_data["ANGLES_INC_HYDROGEN"]
            + self.raw_data["ANGLES_WITHOUT_HYDROGEN"]
        )
        self._angleList = []
        # forceConstConversionFactor = (units.kilocalorie_per_mole/(units.radian*units.radian)).conversion_factor_to(units.kilojoule_per_mole/(units.radian*units.radian))
        for ii in range(0, len(anglePointers), 4):
            if (
                int(anglePointers[ii]) < 0
                or int(anglePointers[ii + 1]) < 0
                or int(anglePointers[ii + 2]) < 0
            ):
                raise Exception(
                    "Found negative angle atom pointers {}".format(
                        (
                            anglePointers[ii],
                            anglePointers[ii + 1],
                            anglePointers[ii + 2],
                        ),
                    )
                )
            iType = int(anglePointers[ii + 3])
            i, k = sorted(
                (int(anglePointers[ii]) // 3 + 1, int(anglePointers[ii + 2]) // 3 + 1)
            )
            self._angleList.append(
                (
                    iType,
                    i,
                    int(anglePointers[ii + 1]) // 3 + 1,
                    k,
                    float(forceConstant[iType - 1]),
                    math.degrees(float(angleEquil[iType - 1])),
                )
            )
        return self._angleList

    def parse_dihedral_params(self):
        """Return list of atom quads, K, phase and periodicity for each dihedral angle"""

        forceConstant = self.raw_data["DIHEDRAL_FORCE_CONSTANT"]
        phase = self.raw_data["DIHEDRAL_PHASE"]
        periodicity = self.raw_data["DIHEDRAL_PERIODICITY"]
        dihedralPointers = (
            self.raw_data["DIHEDRALS_INC_HYDROGEN"]
            + self.raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
        )
        self._dihedralList = []
        # forceConstConversionFactor = (units.kilocalorie_per_mole).conversion_factor_to(units.kilojoule_per_mole)
        for ii in range(0, len(dihedralPointers), 5):
            if int(dihedralPointers[ii]) < 0 or int(dihedralPointers[ii + 1]) < 0:
                raise Exception(
                    "Found negative dihedral atom pointers {}".format(
                        (
                            dihedralPointers[ii],
                            dihedralPointers[ii + 1],
                            dihedralPointers[ii + 2],
                            dihedralPointers[ii + 3],
                        ),
                    )
                )
            iType = int(dihedralPointers[ii + 4])
            i, j, k, l = (
                int(dihedralPointers[ii]) // 3 + 1,
                int(dihedralPointers[ii + 1]) // 3 + 1,
                abs(int(dihedralPointers[ii + 2])) // 3 + 1,
                abs(int(dihedralPointers[ii + 3])) // 3 + 1,
            )
            if j > k:
                i, j, k, l = l, k, j, i
            self._dihedralList.append(
                (
                    iType,
                    i,
                    j,
                    k,
                    l,
                    float(forceConstant[iType - 1]),
                    float(phase[iType - 1]),
                    int(0.5 + float(periodicity[iType - 1])),
                )
            )
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

    def parse_nonbond_params(self, atoms):
        """
        Return list of all rVdw, epsilon pairs for each atom. If off-diagonal
        elements of the Lennard-Jones A and B coefficient matrices are found,
        NbfixPresent exception is raised
        """
        # if self._has_nbfix_terms:
        #     raise Exception('Off-diagonal Lennard-Jones elements found. '
        #                 'Cannot determine LJ parameters for individual atoms.')

        # Check if there are any non-zero HBOND terms
        for x, y in zip(self.raw_data["HBOND_ACOEF"], self.raw_data["HBOND_BCOEF"]):
            if float(x) or float(y):
                raise Exception("10-12 interactions are not supported")
        return_list = []
        # lengthConversionFactor = units.angstrom.conversion_factor_to(units.nanometer)
        # energyConversionFactor = units.kilocalorie_per_mole.conversion_factor_to(units.kilojoule_per_mole)
        numTypes = self.meta["NTYPES"]
        atomTypeIndexes = self.raw_data["ATOM_TYPE_INDEX"]
        [(0, 0) for i in range(numTypes)]
        for iAtom in range(self.meta["NATOM"]):
            index = (numTypes + 1) * (atomTypeIndexes[iAtom] - 1)
            nbIndex = int(self.raw_data["NONBONDED_PARM_INDEX"][index]) - 1
            if nbIndex < 0:
                raise Exception("10-12 interactions are not supported")
            acoef = float(self.raw_data["LENNARD_JONES_ACOEF"][nbIndex])
            bcoef = float(self.raw_data["LENNARD_JONES_BCOEF"][nbIndex])
            try:
                # sigma = (acoef / bcoef) ** (1 / 6.0)
                rMin = (2 * acoef / bcoef) ** (1 / 6.0)
                epsilon = 0.25 * bcoef * bcoef / acoef
            except ZeroDivisionError:
                # sigma = 2.5
                rMin = 1.0
                epsilon = 0.0
            # jichen: unit conversion
            # length: angstrom to namometer
            # epsilon: kcal/mol to kJ/mol
            # rVdw = rMin / 2.0
            # type_parameters[atomTypeIndexes[iAtom] - 1] = (rVdw, epsilon)
            sigma = 2 ** (-1 / 6) * rMin
            return_list.append((iAtom + 1, sigma, epsilon))
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
