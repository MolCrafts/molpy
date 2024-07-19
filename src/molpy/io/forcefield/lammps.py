from pathlib import Path
import molpy as mp

from functools import partial

BOND_TYPE_FIELDS = {
    "harmonic": ["k", "r0"],
}

ANGLE_TYPE_FIELDS = {
    "harmonic": ["k", "theta0"],
    "charmm": ["k", "theta0", "Kub", "rub"],
}

DIHEDRAL_TYPE_FIELDS = {
    "charmm": ["k", "n", "delta", "w"],
    "multi/harmonic": ["k1", "k2", "k3", "k4", "n", "delta"],
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

    def read(self) -> mp.ForceField:

        in_lines: list[str] = []
        data_lines: list[str] = []

        for file in self.files:
            file_path = Path(file)
            with open(file, "r") as f:
                if file_path.name.endswith(".in"):
                    in_lines += f.readlines()
                elif file_path.name.endswith(".data"):
                    data_lines += f.readlines()
                else:
                    raise ValueError(f"Unknown file type: {file_path.name}")

        for i, (line, comment) in enumerate(map(self.clean_line, in_lines)):

            line = line.split()
            if not line:
                continue
            kw = line[0]

            if kw == "units":
                self.forcefield.unit = line[1]

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
            results = {}
            style_ = ""
            i = 1
            while i < len(line):
                if not line[i].isdigit():
                    style_ = line[i]
                    results[style_] = []
                else:
                    results[style_].append(line[i])
                i += 1
            for style, coeffs in results.items():
                self.forcefield.def_dihedralstyle(style)

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

        if line[1].isalpha():  # hybrid
            bondstyle = self.forcefield.get_bondstyle(line[1])
            coeffs = line[2:]
        else:
            bondstyle = self.forcefield.bondstyles[0]
            coeffs = line[1:]

        if bondstyle.name in BOND_TYPE_FIELDS:
            named_params = {
                k: float(v) for k, v in zip(BOND_TYPE_FIELDS[bondstyle.name], coeffs)
            }
            bondstyle.def_bondtype(
                name=bond_type_id,
                **{k: float(v) for k, v in named_params.items()},
            )
        else:
            params = [float(v) for v in coeffs]
            bondstyle.def_bondtype(
                name=bond_type_id,
                *params,
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
                line = line.split()  # Pair Coeffs syntax:
                type_id = line[0]    # i ...
                if type_id.isalpha():
                    break
                self.read_pair_coeff(line)

    def read_angle_coeff(self, line):

        angle_type_id = line[0]

        if line[1].isalpha():  # hybrid
            anglestyle = self.forcefield.get_anglestyle(line[1])
            coeffs = line[2:]
        else:
            anglestyle = self.forcefield.anglestyles[0]
            coeffs = line[1:]

        if anglestyle.name in ANGLE_TYPE_FIELDS:

            anglestyle.def_angletype(
                name=angle_type_id,
                **{k: float(v) for k, v in zip(ANGLE_TYPE_FIELDS[anglestyle.name], coeffs)},
            )
        else:
            anglestyle.def_angletype(
                name=anglestyle.name,
                *[v for v in coeffs],
            )

    def read_dihedral_coeff(self, line):

        dihedral_type_id = line[0]

        if line[1].isdigit():  # hybrid
            dihedralstyle = self.forcefield.get_dihedralstyle(line[1])
            coeffs = line[2:]
        else:
            dihedralstyle = self.forcefield.dihedralstyles[0]
            coeffs = line[1:]

        if dihedralstyle.name in DIHEDRAL_TYPE_FIELDS:

            dihedralstyle.def_dihedraltype(
                name=dihedral_type_id,
                **{
                    k: float(v)
                    for k, v in zip(DIHEDRAL_TYPE_FIELDS[dihedralstyle.name], coeffs)
                },
            )

        else:
            dihedralstyle.def_dihedraltype(
                name=dihedralstyle.name,
                *[v for v in coeffs],
            )

    def read_improper_coeff(self, line):

        improper_type_id = line[0]

        if self.forcefield.n_improperstyles > 1:
            improperstyle = self.forcefield.get_improperstyle(line[1])
            coeffs = line[2:]

        else:
            improperstyle = self.forcefield.improperstyles[0]
            coeffs = line[1:]

        if improperstyle.name in IMPROPER_TYPE_FIELDS:
            improperstyle.def_impropertype(
                improper_type_id,
                **{
                    k: float(v)
                    for k, v in zip(IMPROPER_TYPE_FIELDS[improperstyle.name], coeffs)
                },
            )
        else:
            improperstyle.def_impropertype(
                name=improperstyle.name,
                *[v for v in coeffs],
            )

    def read_pair_coeff(self, line):

        atomtype_i = atomtype_j = int(line[0])

        if len(self.forcefield.pairstyles) > 1:
            pairstyle = self.forcefield.get_pairstyle(line[2])
            coeffs = line[3:]

        else:
            pairstyle = self.forcefield.pairstyles[0]
            coeffs = line[2:]

        name = f"{atomtype_i} {atomtype_j}"

        if pairstyle.name in PAIR_TYPE_FIELDS:
            pairstyle.def_pairtype(
                name,
                atomtype_i,
                atomtype_j,
                **{k: v for k, v in zip(PAIR_TYPE_FIELDS[pairstyle.name], coeffs)},
            )
        else:
            pairstyle.def_pairtype(
                name=name,
                i=atomtype_i,
                j=atomtype_j,
                *[v for v in coeffs],
            )

    def read_pair_modify(self, line):

        if line[0] == "pair":
            raise NotImplementedError("pair_modify hybrid not implemented")
        else:
            assert self.forcefield.n_pairstyles == 1, ValueError(
                "pair_modify command requires one pair style"
            )
            pairstyle = self.forcefield.pairstyles[0]

            if "modified" in pairstyle.named_params:
                pairstyle.named_params["modified"] += line
            else:
                pairstyle.named_params["modified"] = line


class LAMMPSForceFieldWriter:

    def __init__(self, fpath: str | Path, forcefield: mp.ForceField):

        self.fpath = fpath
        self.forcefield = forcefield

    @staticmethod
    def _write_styles(lines: list[str], styles, style_type):

        if len(styles) == 1:
            style = styles[0]
            if len(style.types) == 0:
                return
            lines.append(f"# {style_type}_style {style.name} {' '.join(style.params)}\n")
            if "modified" in style.named_params:
                params = " ".join(style.named_params["modified"])
                lines.append(f"{style_type}_modify {params}\n")

            for typ in style.types:
                params = " ".join(map(str, typ.params))
                named_params = " ".join(typ.named_params.values())
                lines.append(f"{style_type}_coeff {typ.name} {params} {named_params}\n")
        else:
            style_keywords = " ".join([style.name for style in styles])
            lines.append(f"{style_type}_style hybrid {style_keywords}\n")
            for style in styles:
                for typ in style.types:
                    params = " ".join(map(str, typ.params))
                    named_params = " ".join(typ.named_params.values())
                    lines.append(
                        f"{style_type}_coeff {typ.name} {style.name} {params} {named_params}\n"
                    )

        lines.append("\n")

    @staticmethod
    def _write_pair_styles(lines: list[str], styles, style_type:str):

        if len(styles) == 1:
            style = styles[0]
            lines.append(f"# {style_type}_style {style.name} {' '.join(style.params)}\n")
            if "modified" in style.named_params:
                params = " ".join(style.named_params["modified"])
                lines.append(f"{style_type}_modify {params}\n")

            for typ in style.types:
                params = " ".join(map(str, typ.params))
                named_params = " ".join(typ.named_params.values())
                lines.append(f"{style_type}_coeff {' '.join(map(str, typ.type_idx))} {params} {named_params}\n")
        else:
            style_keywords = " ".join([style.name for style in styles])
            lines.append(f"{style_type}_style hybrid {style_keywords}\n")
            for style in styles:
                for typ in style.types:
                    params = " ".join(map(str, typ.params))
                    named_params = " ".join(typ.named_params.values())
                    lines.append(
                        f"{style_type}_coeff {' '.join(map(str, typ.type_idx))} {style.name} {params} {named_params}\n"
                    )

        lines.append("\n")

    def write(self):

        ff = self.forcefield

        lines = []

        # lines.append(f"units {self.forcefield.unit}\n")
        if ff.atomstyles:
            if ff.n_anglestyles == 1:
                lines.append(f"# atom_style {ff.atomstyles[0].name}\n")
            else:
                atomstyles = " ".join([atomstyle.name for atomstyle in ff.atomstyles])
                lines.append(f"atom_style hybrid {atomstyles}\n")
        else:
            raise ValueError("No atom style defined")

        for atomstyle in ff.atomstyles:
            for atomtype in atomstyle.types:
                lines.append(f"mass {atomtype.name} {atomtype.named_params['mass']}\n")

        LAMMPSForceFieldWriter._write_styles(lines, ff.bondstyles, "bond")
        LAMMPSForceFieldWriter._write_styles(lines, ff.anglestyles, "angle")
        LAMMPSForceFieldWriter._write_styles(lines, ff.dihedralstyles, "dihedral")
        if ff.n_impropertypes:
            LAMMPSForceFieldWriter._write_styles(lines, ff.improperstyles, "improper")
        LAMMPSForceFieldWriter._write_pair_styles(lines, ff.pairstyles, "pair")

        lines.append("\n")

        with open(self.fpath, "w") as f:

            f.writelines(lines)
