from pathlib import Path
import molpy as mp

BOND_TYPE_FIELDS = {
    "harmonic": ["k", "r0"],
}

ANGLE_TYPE_FIELDS = {
    "harmonic": ["k", "theta0"],
    "charmm": ["k", "theta0", "Kub", "rub"]
}

DIHEDRAL_TYPE_FIELDS = {
    "multi/harmonic": ["k", "n", "delta"],
    "charmm": ["k1", "k2", "k3", "k4", "n", "delta"],
}

IMPROPER_TYPE_FIELDS = {
    "harmonic": ["k", "chi0"],
}

PAIR_TYPE_FIELDS = {
    "lj/cut": ["epsilon", "sigma"],
    "lj/cut/coul/long": ["epsilon", "sigma"],
    "lj/charmm/coul/long": ["epsilon", "sigma", "eps14", "sig14"],
}

class LAMMPSForceFieldReader:

    def __init__(
        self, files: list[str | Path], forcefield: mp.ForceField | None = None
    ):

        self.files = files
        if forcefield is None:
            self.forcefield = mp.ForceField()
        else:
            self.forcefield = forcefield

    def load(self):

        in_lines: list[str] = []
        data_lines: list[str] = []

        for file in self.files:
            file_path = Path(file)
            with open(file, "r") as f:
                if file_path.name.endswith(".in"):
                    in_lines += f.readlines()
                elif file_path.name.endswith(".data"):
                    data_lines += f.readlines()

        n_atomtypes = 0
        n_bondtypes = 0
        n_angletypes = 0
        n_dihedraltypes = 0
        n_impropertypes = 0

        for i, (line, comment) in enumerate(map(self.clean_line, in_lines)):

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

            elif kw == "mass":
                self.read_mass_line(line[1:], comment)

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

        for i, (line, comment) in enumerate(map(self.clean_line, data_lines)):

            line = line.split()
            if not line:
                continue
            kw = line[0]

            if kw == "Masses":
                self.read_mass_section(data_lines[i + 1 :])

            elif "Coeffs" in line:

                if kw == "Bond":
                    self.read_bond_coeff_section(data_lines[i + 1 :])

                elif kw == "Angle":
                    self.read_angle_coeff_section(data_lines[i + 1 :])

                elif kw == "Dihedral":
                    self.read_dihedral_coeff_section(data_lines[i + 1 :])

                elif kw == "Improper":
                    self.read_improper_coeff_section(data_lines[i + 1 :])

                elif kw == "Pair":
                    self.read_pair_coeff_section(data_lines[i + 1 :])

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

        # assert self.forcefield.n_atomtypes == n_atomtypes, ValueError(
        #     f"Number of atom types mismatch: {self.forcefield.n_atomtypes} != {n_atomtypes}"
        # )
        # assert self.forcefield.n_bondtypes == n_bondtypes, ValueError(
        #     f"Number of bond types mismatch: {self.forcefield.n_bondtypes} != {n_bondtypes}"
        # )
        # assert self.forcefield.n_angletypes == n_angletypes, ValueError(
        #     f"Number of angle types mismatch: {self.forcefield.n_angletypes} != {n_angletypes}"
        # )
        # assert self.forcefield.n_dihedraltypes == n_dihedraltypes, ValueError(
        #     f"Number of dihedral types mismatch: {self.forcefield.n_dihedraltypes} != {n_dihedraltypes}"
        # )
        # assert self.forcefield.n_impropertypes == n_impropertypes, ValueError(
        #     f"Number of improper types mismatch: {self.forcefield.n_impropertypes} != {n_impropertypes}"
        # )
        # assert self.forcefield.n_pairtypes == n_atomtypes * n_atomtypes, ValueError(
        #     f"Number of pair types mismatch: {self.forcefield.n_pairtypes} != {n_atomtypes * n_atomtypes}"
        # )

        return self.forcefield

    def clean_line(self, line: str) -> tuple[str, str]:
        return line.partition("#")[0].strip(), line.partition("#")[-1]

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

    def read_mass_section(self, lines: list[str]):

        for i, (line, comment) in enumerate(map(self.clean_line, lines)):
            line = line.split()
            if line:
                self.read_mass_line(line, comment)
                break

        for line, comment in map(self.clean_line, lines[i + 1 :]):
            line = line.split()
            if line:
                self.read_mass_line(line, comment)
            else:
                break

    def read_mass_line(self, line: list[str], comment: str):
        atomstyle = self.forcefield.atomstyles[0]
        atomtype = atomstyle.get_atomtype(line[0])
        if comment:  # not empty
            name = comment
        else:
            name = line[0]  # same as atom type
        mass = float(line[1])
        atomtype_id = int(line[0])
        if atomtype:
            atomtype["mass"] = mass
        else:
            atomstyle.def_atomtype(name, atomtype_id, mass=mass)

    def read_bond_coeff(self, line):

        bond_type_id = line[0]

        if len(self.forcefield.bondstyles) > 1:
            s = self.forcefield.get_bondstyle(line[1])
            coeffs = line[2:]

        else:
            bondstyle = self.forcefield.bondstyles[0]
            coeffs = line[1:]
        bondstyle.def_bondtype(
            name=bond_type_id,
            **{k: float(v) for k, v in zip(BOND_TYPE_FIELDS[bondstyle.name], coeffs)},
        )

    def read_bond_coeff_section(self, lines: list[str]):

        for line, comment in map(self.clean_line, lines):
            if line:
                line = line.split()
                if line[0].isalpha():
                    break
                self.read_bond_coeff(line)

    def read_angle_coeff_section(self, lines: list[str]):

        for line, comment in map(self.clean_line, lines):
            if line:
                line = line.split()
                if line[0].isalpha():
                    break
                self.read_angle_coeff(line)


    def read_dihedral_coeff_section(self, lines: list[str]):

        for line, comment in map(self.clean_line, lines):
            if line:
                line = line.split()
                if line[0].isalpha():
                    break
                self.read_dihedral_coeff(line)


    def read_improper_coeff_section(self, lines: list[str]):

        for line, comment in map(self.clean_line, lines):
            if line:
                line = line.split()
                type_id = line[0]
                if type_id.isalpha():
                    break
                self.read_improper_coeff(line)


    def read_pair_coeff_section(self, lines: list[str]):

        for line, comment in map(self.clean_line, lines):

            if line and line.isalnum:
                line = line.split()      # Pair Coeffs syntax:
                type_id = line[0]        # i ...
                if type_id.isalpha():
                    break
                line.insert(0, type_id)  # pair_coeff i j ...
                self.read_pair_coeff(line)

    def read_angle_coeff(self, line):

        angle_type_id = line[0]

        if self.forcefield.n_anglestyles > 1:
            anglestyle = self.forcefield.get_anglestyle(line[1])
            coeffs = line[2:]

        else:
            anglestyle = self.forcefield.anglestyles[0]
            coeffs = line[1:]
        anglestyle.def_angletype(
            name=angle_type_id,
            **{
                k: float(v) for k, v in zip(ANGLE_TYPE_FIELDS[anglestyle.name], coeffs)
            },
        )

    def read_dihedral_coeff(self, line):

        dihedral_type_id = line[0]

        if self.forcefield.n_dihedralstyles > 1:
            dihedralstyle = self.forcefield.get_dihedralstyle(line[1])
            coeffs = line[2:]

        else:
            dihedralstyle = self.forcefield.dihedralstyles[0]
            coeffs = line[1:]
        dihedralstyle.def_dihedraltype(
            name=dihedral_type_id,
            **{
                k: float(v)
                for k, v in zip(DIHEDRAL_TYPE_FIELDS[dihedralstyle.name], coeffs)
            },
        )

    def read_improper_coeff(self, line):

        improper_type_id = line[0]

        if self.forcefield.n_improperstyles > 1:
            improperstyle = self.forcefield.get_improperstyle(line[1])
            coeffs = line[2:]

        else:
            improperstyle = self.forcefield.improperstyles[0]
            coeffs = line[1:]
        improperstyle.def_impropertype(
            name=improper_type_id,
            **{
                k: float(v)
                for k, v in zip(IMPROPER_TYPE_FIELDS[improperstyle.name], coeffs)
            },
        )

    def read_pair_coeff(self, line):

        atomtype_i = int(line[0])
        atomtype_j = int(line[1])

        if len(self.forcefield.pairstyles) > 1:
            pairstyle = self.forcefield.get_pairstyle(line[2])
            coeffs = line[3:]

        else:
            pairstyle = self.forcefield.pairstyles[0]
            coeffs = line[2:]

        name = str(pairstyle.n_types)

        pairstyle.def_pairtype(
            name,
            atomtype_i,
            atomtype_j,
            **{k: float(v) for k, v in zip(PAIR_TYPE_FIELDS[pairstyle.name], coeffs)},
        )

    def read_pair_modify(self, line):

        modify = {}
        for k, v in line[::2]:
            modify[k] = v

        for pairstyle in self.forcefield.pairstyles:
            pairstyle.update(**modify)
