import molpy as mp

BOND_TYPE_FIELDS = {
    "harmonic": ["k", "r0"],
}

ANGLE_TYPE_FIELDS = {
    "harmonic": ["k", "theta0"],
}

DIHEDRAL_TYPE_FIELDS = {
    "multi/harmonic": ["k", "n", "delta"],
    "charmm": ["k1", "k2", "k3", "k4", "n", "delta"],
}

PAIR_TYPE_FIELDS = {
    "lj/cut": ["epsilon", "sigma"],
    "lj/cut/coul/long": ["epsilon", "sigma"],
    "lj/charmm/coul/long": ["epsilon", "sigma", "eps14", "sig14"],
}

class LAMMPSForceFieldReader:

    def __init__(self, files:list[str], forcefield:mp.ForceField|None=None):

        self.files = files
        if forcefield is None:
            self.forcefield = mp.ForceField()
        else:
            self.forcefield = forcefield

    def load(self):

        lines:list[str] = []

        for file in self.files:
            with open(file, 'r') as f:
                lines += f.readlines()

        for i, line in enumerate(map(self.clean_line, lines)):
            
            line = line.split()
            if not line:
                continue
            kw = line[0]
            if kw == "units":
                self.forcefield.units = line[1]

            elif kw == "bond_style":
                self.read_bondstyle(line[1:])

            elif kw == "pair_style":
                self.read_pairstyle(line[1:])

            elif kw == "angle_style":
                self.read_anglestyle(line[1:])

            elif kw == "dihedral_style":
                self.read_dihedralstyle(line[1:])

            elif kw == "improper_style":
                self.read_improperstyle(line[1:])

            elif kw == "Masses":
                self.read_mass_section(lines[i+1:])

            elif kw == "mass":
                self.read_mass_line(line[1:])

            elif kw == "bond_coeff":
                self.read_bond_coeff(line[1:])

            elif kw == "angle_coeff":
                self.read_angle_coeff(line[1:])

            elif kw == "dihedral_coeff":
                self.read_dihedral_coeff(line[1:])

            elif kw == "pair_coeff":
                self.read_pair_coeff(line[1:])
            
            elif kw == "pair_modify":
                self.read_pair_modify(line[1:])

            elif kw == "atom_style":
                self.read_atom_style(line[1:])

            elif kw == "Pair Coeffs":
                self.read_pair_coeff_section(lines[i+1:])

            if line[-1] == "types": 

                if line[-2] == "atom":
                    n_atomtypes = int(line[0])

                elif line[-2] == "bond":
                    n_bondtypes = int(line[0])

                elif line[-2] == "angle":
                    n_angletypes = int(line[0])

                elif line[-2] == "dihedral":
                    n_dihedraltypes = int(line[0])

                elif line[-2] == "improper":
                    n_impropertypes = int(line[0])

        assert self.forcefield.n_atomtypes == n_atomtypes
        assert self.forcefield.n_bondtypes == n_bondtypes
        assert self.forcefield.n_angletypes == n_angletypes
        assert self.forcefield.n_dihedraltypes == n_dihedraltypes
        assert self.forcefield.n_impropertypes == n_impropertypes
        assert self.forcefield.n_pairtypes == n_atomtypes * n_atomtypes

        return self.forcefield

    def clean_line(self, line):
        return line.partition("#")[0]
    
    def read_atom_style(self, line):
        self.forcefield.def_atomstyle(line[0])
            
    def read_bondstyle(self, line):
        
        if line[0] == "hybrid":
            self.read_bondstyle(line[1:])

        else:
            self.forcefield.def_bondstyle(line[0])

    def read_anglestyle(self, line):

        if line[0] == "hybrid":
            self.read_anglestyle(line[1:])

        else:
            self.forcefield.def_anglestyle(line[0])

    def read_dihedralstyle(self, line):

        if line[0] == "hybrid":
            self.read_dihedralstyle(line[1:])

        else:
            self.forcefield.def_dihedralstyle(line[0])

    def read_improperstyle(self, line):

        if line[0] == "hybrid":
            self.read_improperstyle(line[1:])

        else:
            self.forcefield.def_improperstyle(line[0])

    def read_pairstyle(self, line):
        
        if line[0] == "hybrid":
            self.read_pairstyle(line[1:])

        else:
            self.forcefield.def_pairstyle(line[0])

    def read_mass_section(self, lines:list[str]):

        for i, line in enumerate(map(self.clean_line, lines)):
            line = line.split()
            if line:
                self.read_mass_line(line)
                break

        for line in map(self.clean_line, lines[i+1:]):
            line = line.split()
            if line:
                self.read_mass_line(line)
            else:
                break

    def read_mass_line(self, line):
        atomstyle = self.forcefield.atomstyles[0]
        atomtype = atomstyle.get_atomtype(line[0])
        mass = float(line[1])
        if atomtype:
            atomtype["mass"] = mass
        else:
            atomstyle.def_atomtype(line[0], mass=mass)

    def read_bond_coeff(self, line):

        bond_type_id = int(line[0])

        if len(self.forcefield.bondstyle) > 1:
            bondstyle = self.forcefield.get_bondstyle(line[1])
            coeffs = line[2:]

        else:
            bondstyle = self.forcefield.bondstyles[0]
            coeffs = line[1:]
        bondstyle.def_bondtype(bond_type_id, **{k: float(v) for k, v in zip(BOND_TYPE_FIELDS[bondstyle.style], coeffs)})

    def read_bond_ceoff_section(self, lines:list[str]):

        for lin in map(self.clean_line, lines):
            line = line.split()
            if line:
                bond_type_id = line[0]
                coeff = {k: float(v) for k, v in zip(BOND_TYPE_FIELDS[self.forcefield.bondstyles[0].style], line[1:])}
                self.forcefield.bondstyles[0].def_bondtype(bond_type_id, **coeff)

    def read_angle_coeff(self, line):
        
        angle_type_id = int(line[0])

        if len(self.forcefield.anglestyle) > 1:
            anglestyle = self.forcefield.get_anglestyle(line[1])
            coeffs = line[2:]

        else:
            anglestyle = self.forcefield.anglestyles[0]
            coeffs = line[1:]
        anglestyle.def_angletype(angle_type_id, **{k: float(v) for k, v in zip(ANGLE_TYPE_FIELDS[anglestyle.style], coeffs)})

    def read_dihedral_coeff(self, line):
        
        dihedral_type_id = int(line[0])

        if len(self.forcefield.dihedralstyle) > 1:
            dihedralstyle = self.forcefield.get_dihedralstyle(line[1])
            coeffs = line[2:]

        else:
            dihedralstyle = self.forcefield.dihedralstyles[0]
            coeffs = line[1:]
        dihedralstyle.def_dihedraltype(dihedral_type_id, **{k: float(v) for k, v in zip(DIHEDRAL_TYPE_FIELDS[dihedralstyle.style], coeffs)})

    def read_pair_coeff(self, line):
        
        atomtype_i = int(line[0])
        atomtype_j = int(line[1])

        if len(self.forcefield.pairstyle) > 1:
            pairstyle = self.forcefield.get_pairstyle(line[2])
            coeffs = line[3:]

        else:
            pairstyle = self.forcefield.pairstyles[0]
            coeffs = line[2:]

        pairstyle.def_pairtype(len(pairstyle.n_types), atomtype_i, atomtype_j, **{k: float(v) for k, v in zip(PAIR_TYPE_FIELDS[pairstyle.style], coeffs)})

    def read_pair_coeff_section(self, lines:list[str]):

        # start from non-empty line
        for line in map(self.clean_line, lines):
            line = line.split()
            if line:
                atomtype_i = line[0]
                coeff = {k: float(v) for k, v in zip(PAIR_TYPE_FIELDS[self.forcefield.pairstyles[0].style], line[1:])}
                self.forcefield.pairstyles[0].def_pairtype(atomtype_i, atomtype_i, atomtype_i, **coeff)

        
    def read_pair_modify(self, line):
        
        modify = {}
        for k, v in line[::2]:
            modify[k] = v

        for pairstyle in self.forcefield.pairstyles:
            pairstyle.update(**modify)