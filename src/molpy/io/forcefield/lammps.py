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
        self, files: list[str | Path], forcefield: mp.ForceField | None = None
    ):

        self.files = files
        if forcefield is None:
            self.forcefield = mp.ForceField()
        else:
            self.forcefield = forcefield

    def read(self) -> mp.ForceField:

        lines: list[str] = []

        for file in self.files:
            with open(file, "r") as f:
                lines.extend(f.readlines())
        lines = filter(lambda line: line, map(LAMMPSForceFieldReader.sanitizer, lines))
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

            elif kw == "Masses":
                self.read_mass_section(islice(lines, n_atomtypes))

            elif "Coeffs" in line:

                if kw == "Bond":
                    self.read_bond_coeff_section(islice(lines, n_bondtypes))

                elif kw == "Angle":
                    self.read_angle_coeff_section(islice(lines, n_angletypes))

                elif kw == "Dihedral":
                    self.read_dihedral_coeff_section(islice(lines, n_dihedraltypes))

                elif kw == "Improper":
                    self.read_improper_coeff_section(islice(lines, n_impropertypes))

                elif kw == "Pair":
                    self.read_pair_coeff_section(islice(lines, n_pairtypes))

            if line[-1] == "types":

                if line[-2] == "atom":
                    n_atomtypes = int(line[0])
                    self.forcefield.def_atomstyle("")

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

        return self.forcefield

    @staticmethod
    def sanitizer(line: str) -> str:
        return line.split()

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

    def read_mass_section(self, lines):

        for line in lines:
            self.read_mass_line(line)

    def read_mass_line(self, line: list[str]):
        atomstyle = self.forcefield.atomstyles[0]
        atomtype = atomstyle.get_type(line[0])
        mass = float(line[1])
        atomtype_id = int(line[0])
        if atomtype:
            atomtype["mass"] = mass
        else:
            atomstyle.def_type(str(atomtype_id), atomtype_id, mass=mass)

    def read_bond_coeff_section(self, lines: Iterator[str]):

        for line in lines:
            self.read_bond_coeff(line)

    def read_angle_coeff_section(self, lines: Iterator[str]):

        for line in lines:
            self.read_angle_coeff(line)

    def read_dihedral_coeff_section(self, lines: Iterator[str]):

        for line in lines:
            self.read_dihedral_coeff(line)

    def read_improper_coeff_section(self, lines: Iterator[str]):

        for line in lines:
            type_id = line[0]
            if type_id.isalpha():
                break
            self.read_improper_coeff(line)

    def read_pair_coeff_section(self, lines: Iterator[str]):

        for line in lines:
            if line.isalnum:  # Pair Coeffs syntax:
                type_id = line[0]  # i ...
                if type_id.isalpha():
                    break
                line.insert(0, type_id)  # pair_coeff i j ...
                self.read_pair_coeff(line)


    def read_bond_coeff(self, line):

        bond_type_id = line[0]

        if line[1].isalpha():  # hybrid
            bondstyle = self.forcefield.get_bondstyle(line[1])
            bondstyle_name = line[1]
            coeffs = line[2:]
        else:
            bondstyle = self.forcefield.get_bondstyle(bond_type_id)
            bondstyle_name = bond_type_id
            coeffs = line[1:]

        if bondstyle is None:
            bondstyle = self.forcefield.def_bondstyle(bondstyle_name)

        if bondstyle.name in BOND_TYPE_FIELDS:
            named_params = {
                k: v for k, v in zip(BOND_TYPE_FIELDS[bondstyle.name], coeffs)
            }
            bondstyle.def_type(
                bond_type_id,
                **named_params,
            )
        else:
            params = [v for v in coeffs]
            bondstyle.def_type(
                bond_type_id,
                *params,
            )

    def read_angle_coeff(self, line):

        angle_type_id = line[0]

        if line[1].isalpha():  # hybrid
            anglestyle = self.forcefield.get_anglestyle(line[1])
            anglestyle_name = line[1]
            coeffs = line[2:]
        else:
            anglestyle = self.forcefield.get_anglestyle(angle_type_id)
            anglestyle_name = angle_type_id
            coeffs = line[1:]
            
        if anglestyle is None:
            anglestyle = self.forcefield.def_anglestyle(anglestyle_name)

        if anglestyle.name in ANGLE_TYPE_FIELDS:

            anglestyle.def_type(
                angle_type_id,
                **{k: v for k, v in zip(ANGLE_TYPE_FIELDS[anglestyle.name], coeffs)},
            )
        else:
            anglestyle.def_type(
                anglestyle.name,
                *[v for v in coeffs],
            )

    def read_dihedral_coeff(self, line):

        dihedral_type_id = line[0]

        if line[1].isalpha():  # hybrid
            dihedralstyle = self.forcefield.get_dihedralstyle(line[1])
            dihedralsyle_name = line[1]
            coeffs = line[2:]
        else:
            dihedralstyle = self.forcefield.get_dihedralstyle(dihedral_type_id)
            dihedralsyle_name = dihedral_type_id
            coeffs = line[1:]

        if dihedralstyle is None:
            dihedralstyle = self.forcefield.def_dihedralstyle(dihedralsyle_name)

        if dihedralstyle.name in DIHEDRAL_TYPE_FIELDS:

            dihedralstyle.def_type(
                dihedral_type_id,
                **{
                    k: v
                    for k, v in zip(DIHEDRAL_TYPE_FIELDS[dihedralstyle.name], coeffs)
                },
            )

        else:
            dihedralstyle.def_type(
                dihedralstyle.name, None, None, None, None, *[v for v in coeffs],
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
            improperstyle.def_type(
                improper_type_id,
                **{
                    k: v
                    for k, v in zip(IMPROPER_TYPE_FIELDS[improperstyle.name], coeffs)
                },
            )
        else:
            improperstyle.def_type(
                improperstyle.name,
                *[v for v in coeffs],
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

        if pairstyle.name in PAIR_TYPE_FIELDS:
            pairstyle.def_pairtype(
                name,
                atomtype_i,
                atomtype_j,
                **{k: v for k, v in zip(PAIR_TYPE_FIELDS[pairstyle.name], coeffs)},
            )
        else:
            pairstyle.def_pairtype(
                name,
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
                pairstyle.named_params["modified"] = list(
                    set(pairstyle.named_params["modified"]) | set(line)
                )
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
