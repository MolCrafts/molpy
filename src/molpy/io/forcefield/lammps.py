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
}

class LAMMPSForceField:

    def __init__(self, files:list[str], forcefield:mp.ForceField|None=None):

        self.files = files
        self.forcefield = forcefield

    def load(self):

        lines:list[str] = []
        if self.forcefield:
            forcefield = self.forcefield
        else:
            forcefield = mp.ForceField()
        forcefield.def_atom_style("full")

        for file in self.files:
            with open(file, 'r') as f:
                lines += f.readlines()

        for line in map(lines, self.clean_line):

            line = line.split()

            if line[0] == "bond_style":
                self.read_bond_style(line[1:], forcefield)

            elif line[0] == "pair_style":
                self.read_pair_style(line[1:], forcefield)

            elif line[0] == "mass":
                self.read_mass(line[1:], forcefield)

            elif line[0] == "bond_coeff":
                self.read_bond_coeff(line[1:], forcefield)

            elif line[0] == "angle_coeff":
                self.read_angle_coeff(line[1:], forcefield)

            elif line[0] == "dihedral_coeff":
                self.read_dihedral_coeff(line[1:], forcefield)

            elif line[0] == "pair_coeff":
                self.read_pair_coeff(line[1:], forcefield)
            
            elif line[0] == "pair_modify":
                self.read_pair_modify(line[1:], forcefield)

    def clean_line(self, line):
        return line.split("#")[0].split()
            
    def read_bond_style(self, line, forcefield):
        
        if line[0] == "hybrid":
            self.read_bond_style(line[1:], forcefield)

        else:
            forcefield.def_bond_style(line[0], line[1:])

    def read_pair_style(self, line, forcefield):
        
        if line[0] == "hybrid":
            self.read_pair_style(line[1:], forcefield)

        else:
            forcefield.def_pair_style(line[0], line[1:])

    def read_mass(self, line, forcefield):
        atom_style = forcefield.atom_styles[0]
        atom_type = atom_style.get_atom_type(line[0])
        if atom_type:
            atom_type["mass"] = float(line[1])
        else:
            atom_style.def_atom_type(line[0], mass=float(line[1]))

    def read_bond_coeff(self, line, forcefield):

        bond_type_id = int(line[0])

        if len(forcefield.bond_style) > 1:
            bond_style = forcefield.get_bond_style(line[1])
            coeffs = line[2:]

        else:
            bond_style = forcefield.bond_styles[0]
            coeffs = line[1:]
        bond_style.def_bond_type(bond_type_id, **{k: float(v) for k, v in zip(BOND_TYPE_FIELDS[bond_style.style], coeffs)})

    def read_angle_coeff(self, line, forcefield):
        
        angle_type_id = int(line[0])

        if len(forcefield.angle_style) > 1:
            angle_style = forcefield.get_angle_style(line[1])
            coeffs = line[2:]

        else:
            angle_style = forcefield.angle_styles[0]
            coeffs = line[1:]
        angle_style.def_angle_type(angle_type_id, **{k: float(v) for k, v in zip(ANGLE_TYPE_FIELDS[angle_style.style], coeffs)})

    def read_dihedral_coeff(self, line, forcefield):
        
        dihedral_type_id = int(line[0])

        if len(forcefield.dihedral_style) > 1:
            dihedral_style = forcefield.get_dihedral_style(line[1])
            coeffs = line[2:]

        else:
            dihedral_style = forcefield.dihedral_styles[0]
            coeffs = line[1:]
        dihedral_style.def_dihedral_type(dihedral_type_id, **{k: float(v) for k, v in zip(DIHEDRAL_TYPE_FIELDS[dihedral_style.style], coeffs)})

    def read_pair_coeff(self, line, forcefield):
        
        atom_type_i = int(line[0])
        atom_type_j = int(line[1])

        if len(forcefield.pair_style) > 1:
            pair_style = forcefield.get_pair_style(line[2])
            coeffs = line[3:]

        else:
            pair_style = forcefield.pair_styles[0]
            coeffs = line[2:]

        pair_style.def_pair_type(len(pair_style.n_types), atom_type_i, atom_type_j, **{k: float(v) for k, v in zip(PAIR_TYPE_FIELDS[pair_style.style], coeffs)})
        
    def read_pair_modify(self, line, forcefield):
        
        modify = {}
        for k, v in line[::2]:
            modify[k] = v

        for pair_style in forcefield.pair_styles:
            pair_style.update(**modify)