from pathlib import Path
import molpy as mp
from itertools import islice
from typing import Iterator

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
        self, file: Path
    ):

        self.file = file

    def read(self, system) -> mp.ForceField:

        self.forcefield: mp.ForceField = system.forcefield
        with open(self.file, "r") as f:
            lines = f.readlines()
        lines = filter(lambda line: line, map(LAMMPSForceFieldReader.sanitizer, lines))
        # default values
        # bondstyle_name = "unknown"
        # anglestyle_name = "unknown"
        # dihedralstyle_name = "unknown"
        # improperstyle_name = "unknown"
        # pairstyle_name = "unknown"
        n_pairtypes = 0
        for line in lines:
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
                self.mass_per_atomtype = self.read_mass_line(line[1:])

            elif kw == "bond_coeff":
                self.read_bond_coeff(self.bondstyle, line[1:])

            elif kw == "angle_coeff":
                self.read_angle_coeff(self.anglestyle, line[1:])

            elif kw == "dihedral_coeff":
                self.read_dihedral_coeff(self.dihedralstyle, line[1:])

            elif kw == "pair_coeff":
                self.read_pair_coeff(self.pairstyle, line[1:])

            elif kw == "pair_modify":
                self.read_pair_modify(line[1:])

            elif kw == "atom_style":
                self.read_atom_style(line[1:])

            elif kw == "Masses":
                self.read_mass_section(islice(lines, n_atomtypes))

            elif "Coeffs" in line:

                if kw == "Bond":
                    if "#" in line:
                        bondstyle_name = line[line.index("#") + 1]
                    self.read_bond_coeff_section(
                        bondstyle_name, islice(lines, n_bondtypes)
                    )

                elif kw == "Angle":
                    if "#" in line:
                        anglestyle_name = line[line.index("#") + 1]
                    self.read_angle_coeff_section(
                        anglestyle_name, islice(lines, n_angletypes)
                    )

                elif kw == "Dihedral":
                    if "#" in line:
                        dihedralstyle_name = line[line.index("#") + 1]
                    self.read_dihedral_coeff_section(
                        dihedralstyle_name, islice(lines, n_dihedraltypes)
                    )

                elif kw == "Improper":
                    if "#" in line:
                        improperstyle_name = line[line.index("#") + 1]
                    self.read_improper_coeff_section(
                        improperstyle_name, islice(lines, n_impropertypes)
                    )

                elif kw == "Pair":
                    if "#" in line:
                        pairstyle_name = line[line.index("#") + 1]
                    self.read_pair_coeff_section(
                        pairstyle_name, islice(lines, n_pairtypes)
                    )

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

                elif line[-2] == "pair":
                    n_pairtypes = int(line[0])

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

        return system

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.split()

    def read_atom_style(self, line):
        self.atomstyle = self.forcefield.def_atomstyle(line[0])

    def read_bondstyle(self, line):

        if line[0] == "hybrid":
            self.read_bondstyle(line[1:])

        else:
            self.bondstyle = self.forcefield.def_bondstyle(line[0])

    def read_anglestyle(self, line):

        if line[0] == "hybrid":
            self.read_anglestyle(line[1:])

        else:
            self.anglestyle = self.forcefield.def_anglestyle(line[0])

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
            self.dihedralstyle =self.forcefield.def_dihedralstyle(line[0])

    def read_improperstyle(self, line):

        if line[0] == "hybrid":
            self.read_improperstyle(line[1:])

        else:
            self.improperstyle = self.forcefield.def_improperstyle(line[0])

    def read_pairstyle(self, line):

        if line[0] == "hybrid":
            self.read_pairstyle(line[1:])

        else:
            self.pairstyle = self.forcefield.def_pairstyle(line[0])

    def read_mass_section(self, lines):
        masses = {}
        for line in lines:
            i, m = self.read_mass_line(line)
            masses[i] = m

    def read_mass_line(self, line: list[str]):
        return int(line[0]), float(line[1])

    def read_bond_coeff_section(self, stylename: str, lines: Iterator[str]):
        bondstyle = self.forcefield.get_bondstyle(stylename)
        if bondstyle is None:
            bondstyle = self.forcefield.def_bondstyle(stylename)
        for line in lines:
            self.read_bond_coeff(bondstyle, line)

    def read_angle_coeff_section(self, stylename: str, lines: Iterator[str]):
        anglestyle = self.forcefield.get_anglestyle(stylename)
        if anglestyle is None:
            anglestyle = self.forcefield.def_anglestyle(stylename)
        for line in lines:
            self.read_angle_coeff(anglestyle, line)

    def read_dihedral_coeff_section(self, stylename: str, lines: Iterator[str]):
        dihedralstyle = self.forcefield.get_dihedralstyle(stylename)
        if dihedralstyle is None:
            dihedralstyle = self.forcefield.def_dihedralstyle(stylename)
        for line in lines:
            self.read_dihedral_coeff(dihedralstyle, line)

    def read_improper_coeff_section(self, stylename: str, lines: Iterator[str]):
        improperstyle = self.forcefield.get_improperstyle(stylename)
        if improperstyle is None:
            improperstyle = self.forcefield.def_improperstyle(stylename)
        for line in lines:
            type_id = line[0]
            if type_id.isalpha():
                break
            self.read_improper_coeff(improperstyle, line)

    def read_pair_coeff_section(self, stylename: str, lines: Iterator[str]):
        pairstyle = self.forcefield.get_pairstyle(stylename)
        if pairstyle is None:
            pairstyle = self.forcefield.def_pairstyle(stylename)
        for line in lines:
            if line[0].isalpha():
                break
            line.insert(0, line[0])  # pair_coeff i j ...
            self.read_pair_coeff(pairstyle, line)

    def read_bond_coeff(self, style, line):

        bond_type_id = int(line[0])

        if line[1].isalpha():  # hybrid
            bondstyle_name = line[1]
            style = self.forcefield.get_bondstyle(bondstyle_name)
            if style is None:
                style = self.forcefield.def_bondstyle(bondstyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        if style.name in BOND_TYPE_FIELDS:
            named_params = {k: v for k, v in zip(BOND_TYPE_FIELDS[style.name], coeffs)}
            style.def_type(
                bond_type_id,
                None, None, 
                kw_params=named_params,
            )
        else:
            style.def_type(
                bond_type_id,
                None, None,
                order_params=coeffs,
            )

    def read_angle_coeff(self, style, line):

        angle_type_id = int(line[0])

        if line[1].isalpha():  # hybrid
            anglestyle_name = line[1]
            style = self.forcefield.get_anglestyle(anglestyle_name)
            if style is None:
                style = self.forcefield.def_anglestyle(anglestyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        if style.name in ANGLE_TYPE_FIELDS:

            style.def_type(
                angle_type_id,
                None, None, None,
                kw_params={k: v for k, v in zip(ANGLE_TYPE_FIELDS[style.name], coeffs)},
            )
        else:
            style.def_type(
                angle_type_id,
                None, None, None,
                order_params=coeffs,
            )

    def read_dihedral_coeff(self, style, line):

        dihedral_type_id = int(line[0])

        if line[1].isalpha():  # hybrid
            dihedralsyle_name = line[1]
            style = self.forcefield.get_dihedralstyle(dihedralsyle_name)
            if style is None:
                style = self.forcefield.def_dihedralstyle(dihedralsyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        if style.name in DIHEDRAL_TYPE_FIELDS:

            style.def_type(
                dihedral_type_id,
                None, None, None, None,
                kw_params={k: v for k, v in zip(DIHEDRAL_TYPE_FIELDS[style.name], coeffs)},
            )

        else:
            style.def_type(
                dihedral_type_id,
                None,
                None,
                None,
                None,
                order_params=coeffs,
            )

    def read_improper_coeff(self, style, line):

        improper_type_id = int(line[0])

        if line[1].isalpha():  # hybrid
            improperstyle_name = line[1]
            style = self.forcefield.get_improperstyle(improperstyle_name)
            if style is None:
                style = self.forcefield.def_improperstyle(improperstyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        if style.name in IMPROPER_TYPE_FIELDS:
            style.def_type(
                improper_type_id,
                None, None, None, None,
                kw_params={k: v for k, v in zip(IMPROPER_TYPE_FIELDS[style.name], coeffs)},
            )
        else:
            style.def_type(
                improper_type_id,
                None,
                None,
                None,
                None,
                order_params=coeffs,
            )

    def read_pair_coeff(self, style, line):

        i, j = int(line[0]), int(line[1])

        if line[2].isalpha():  # hybrid
            pairstyle_name = line[2]
            style = self.forcefield.get_pairstyle(pairstyle_name)
            if style is None:
                style = self.forcefield.def_pairstyle(pairstyle_name)
            coeffs = line[3:]
        else:
            coeffs = line[2:]

        if style.name in PAIR_TYPE_FIELDS:
            style.def_type(
                style.n_types+1,
                None, None,
                kw_params={k: v for k, v in zip(PAIR_TYPE_FIELDS[style.name], coeffs)},
            )
        else:
            style.def_type(
                style.n_types+1,
                None, None,
                order_params=coeffs,
            )

    def read_pair_modify(self, line):

        if line[0] == "pair":
            raise NotImplementedError("pair_modify hybrid not implemented")
        else:
            assert self.forcefield.n_pairstyles == 1, ValueError(
                "pair_modify command requires one pair style"
            )
            pairstyle = self.forcefield.pairstyles[0]

            if "modified" in pairstyle:
                pairstyle["modified"] = list(
                    set(pairstyle["modified"]) | set(line)
                )
            else:
                pairstyle["modified"] = line


class LAMMPSForceFieldWriter:

    def __init__(self, fpath: str | Path, forcefield: mp.ForceField):

        self.fpath = fpath
        self.forcefield = forcefield

    @staticmethod
    def _write_styles(lines: list[str], styles, style_type):

        if len(styles) == 1:
            style = styles[0]
            lines.append(f"{style_type}_style {style.name} {' '.join(style.params)}\n")
            if "modified" in style.named_params:
                params = " ".join(style.named_params["modified"])
                lines.append(f"{style_type}_modify {params}\n")

            for typ in style.types:
                params = " ".join(typ.params)
                named_params = " ".join(typ.named_params.values())
                lines.append(f"{style_type}_coeff {typ.name} {params} {named_params}\n")
        else:
            style_keywords = " ".join([style.name for style in styles])
            lines.append(f"{style_type}_style hybrid {style_keywords}\n")
            for style in styles:
                for typ in style.types:
                    params = " ".join(typ.params)
                    named_params = " ".join(typ.named_params.values())
                    lines.append(
                        f"{style_type}_coeff {typ.name} {style.name} {params} {named_params}\n"
                    )

        lines.append("\n")

    def write(self):

        ff = self.forcefield

        lines = []

        lines.append(f"units {self.forcefield.unit}\n")
        if ff.atomstyles:
            if ff.n_anglestyles == 1:
                lines.append(f"atom_style {ff.atomstyles[0].name}\n")
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
        LAMMPSForceFieldWriter._write_styles(lines, ff.improperstyles, "improper")
        LAMMPSForceFieldWriter._write_styles(lines, ff.pairstyles, "pair")

        lines.append("\n")

        with open(self.fpath, "w") as f:

            f.writelines(lines)
