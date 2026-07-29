from itertools import islice
from pathlib import Path
from typing import TextIO, cast

from molpy import (
    AngleStyle,
    AtomisticForcefield,
    AtomStyle,
    AtomType,
    BondStyle,
    DihedralStyle,
    ForceField,
    ImproperStyle,
    PairStyle,
)
from molpy.core.fields import ForceFieldFormatter
from molpy.core.forcefield import (
    PairCoulTTStyle,
    PairTholeStyle,
)
from molpy.io.data.lammps import LammpsFieldFormatter


class LAMMPSForceFieldReader:
    def __init__(self, scripts: Path | list[Path], data: Path):
        self.scripts = scripts if isinstance(scripts, list) else [scripts]
        self.data = data

    # Helper methods to bridge old and new API
    def _get_style_by_name(self, style_class: type, name: str):
        """Get a style by name, returns None if not found."""
        styles = self.forcefield.get_styles(style_class)
        for style in styles:
            if style.name == name:
                return style
        return None

    def _get_first_atomstyle(self) -> AtomStyle | None:
        """Get the first atom style, or None if no styles exist."""
        styles = self.forcefield.get_styles(AtomStyle)
        return styles[0] if styles else None

    def _find_atomtype_by_name(
        self, atomstyle: AtomStyle, name: str
    ) -> AtomType | None:
        """Find an atom type by name within a style."""
        for atom_type in atomstyle.types:
            if atom_type.name == name:
                return atom_type
        return None

    def _ensure_atomtype(self, name: str) -> AtomType:
        """Ensure an atom type exists, create if needed."""
        atomstyle = self._get_first_atomstyle()
        if not atomstyle:
            # Create default atom style if none exists
            atomstyle = self.forcefield.def_atomstyle("full")

        atomtype = self._find_atomtype_by_name(atomstyle, name)
        if atomtype is None:
            atomtype = atomstyle.def_type(name)
        return atomtype

    def _parse_type_name(self, type_name: str, separator: str = "-") -> list[str]:
        """Parse a composite type name into atom type names.

        Examples:
            "CT-CT" -> ["CT", "CT"]
            "CT-CT-OS" -> ["CT", "CT", "OS"]
            "opls_135" -> ["opls_135"] (no separator)
        """
        if separator in type_name:
            return type_name.split(separator)
        else:
            return [type_name]

    def read(self, forcefield: ForceField) -> ForceField:
        # LAMMPSForceFieldReader expects an AtomisticForcefield
        # Cast to the specific type for proper method access
        self.forcefield: AtomisticForcefield = cast(AtomisticForcefield, forcefield)
        lines = []
        for script in self.scripts:
            with open(script) as f:
                lines.extend(f.readlines())
        with open(self.data) as f:
            lines.extend(f.readlines())
        lines = filter(lambda line: line, map(LAMMPSForceFieldReader.sanitizer, lines))
        n_pairtypes = 0
        n_atomtypes = 0
        n_bondtypes = 0
        n_angletypes = 0
        n_dihedraltypes = 0
        n_impropertypes = 0
        for line in lines:
            kw = line[0]

            if kw == "units":
                forcefield.units = line[1]

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
                self.read_bondcoeff(self.bondstyle, line[1:])

            elif kw == "angle_coeff":
                self.read_angle_coeff(self.anglestyle, line[1:])

            elif kw == "dihedral_coeff":
                self.read_dihedral_coeff(self.dihedralstyle, line[1:])

            elif kw == "pair_coeff":
                self.read_pair_coeff(self.pairstyle, line[1:])

            elif kw == "pair_modify":
                self.read_pair_modify(line[1:])

            elif kw == "atom_style":
                self.read_atomstyle(line[1:])

            # define in data
            elif kw == "Masses":
                self.read_mass_section(islice(lines, n_atomtypes))

            elif "Coeffs" in line:
                if kw == "Bond":
                    if "#" in line:
                        bondstyle_name = line[line.index("#") + 1]
                    else:
                        bondstyle_name = self.bondstyle.name
                    self.read_bondcoeff_section(
                        bondstyle_name, islice(lines, n_bondtypes)
                    )

                elif kw == "Angle":
                    if "#" in line:
                        anglestyle_name = line[line.index("#") + 1]
                    else:
                        anglestyle_name = self.anglestyle.name
                    self.read_angle_coeff_section(
                        anglestyle_name, islice(lines, n_angletypes)
                    )

                elif kw == "Dihedral":
                    if "#" in line:
                        dihedralstyle_name = line[line.index("#") + 1]
                    else:
                        if isinstance(self.dihedralstyle, list):
                            dihedralstyle_name = ""
                        else:
                            dihedralstyle_name = self.dihedralstyle.name
                    self.read_dihedral_coeff_section(
                        dihedralstyle_name, islice(lines, n_dihedraltypes)
                    )

                elif kw == "Improper":
                    if "#" in line:
                        improperstyle_name = line[line.index("#") + 1]
                    else:
                        improperstyle_name = self.improperstyle.name
                    self.read_improper_coeff_section(
                        improperstyle_name, islice(lines, n_impropertypes)
                    )

                elif kw == "Pair":
                    if "#" in line:
                        pairstyle_name = line[line.index("#") + 1]
                    else:
                        pairstyle_name = self.pairstyle.name
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

        return self.forcefield

    @staticmethod
    def sanitizer(line: str) -> list[str]:
        return line.split()

    def read_atomstyle(self, line):
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
            for style, _coeffs in results.items():
                self.forcefield.def_dihedralstyle(style)
            # Store all dihedral styles (hybrid case)
            self.dihedralstyle = self.forcefield.get_styles(DihedralStyle)

        else:
            self.dihedralstyle = self.forcefield.def_dihedralstyle(line[0])

    def read_improperstyle(self, line):
        if line[0] == "hybrid":
            self.read_improperstyle(line[1:])

        else:
            self.improperstyle = self.forcefield.def_improperstyle(line[0])

    def read_pairstyle(self, line):
        if line[0] == "hybrid":
            self.read_pairstyle(line[1:])

        else:
            self.pairstyle = self.forcefield.def_pairstyle(line[0], *line[1:])

    def read_mass_section(self, lines):
        for line in lines:
            type_, m = self.read_mass_line(line)
            atomstyle = self._get_first_atomstyle()
            if atomstyle:
                atom_type = self._find_atomtype_by_name(atomstyle, str(type_))
                if atom_type:
                    atom_type["mass"] = m

    def read_mass_line(self, line: list[str]):
        return line[0], float(line[1])

    def read_bondcoeff_section(self, stylename: str, lines: islice):
        bondstyle = self._get_style_by_name(BondStyle, stylename)
        if bondstyle is None:
            bondstyle = self.forcefield.def_bondstyle(stylename)
        for line in lines:
            self.read_bondcoeff(bondstyle, line)

    def read_angle_coeff_section(self, stylename: str, lines: islice):
        anglestyle = self._get_style_by_name(AngleStyle, stylename)
        if anglestyle is None:
            anglestyle = self.forcefield.def_anglestyle(stylename)
        for line in lines:
            self.read_angle_coeff(anglestyle, line)

    def read_dihedral_coeff_section(self, stylename: str, lines: islice):
        if stylename is not None:
            dihedralstyle = self._get_style_by_name(DihedralStyle, stylename)
            if dihedralstyle is None:
                dihedralstyle = self.forcefield.def_dihedralstyle(stylename)
        else:
            dihedralstyle = None
        for line in lines:
            self.read_dihedral_coeff(dihedralstyle, line)

    def read_improper_coeff_section(self, stylename: str, lines: islice):
        improperstyle = self._get_style_by_name(ImproperStyle, stylename)
        if improperstyle is None:
            improperstyle = self.forcefield.def_improperstyle(stylename)
        for line in lines:
            type_id = line[0]
            if type_id.isalpha():
                break
            self.read_improper_coeff(improperstyle, line)

    def read_pair_coeff_section(self, stylename: str, lines: islice):
        pairstyle = self._get_style_by_name(PairStyle, stylename)
        if pairstyle is None:
            pairstyle = self.forcefield.def_pairstyle(stylename)
        for line in lines:
            # if line[0].isalpha():
            #     break
            # line.insert(0, line[0])  # pair_coeff i j ...
            self.read_pair_coeff(pairstyle, line)

    def read_bondcoeff(self, style, line):
        """Read bond_coeff line and create BondType.

        Format: bond_coeff <type_name> [style_name] <k> <r0>
        Example: bond_coeff CT-CT 268.0 1.529
        """
        bond_type_name = line[0]

        if line[1].isalpha():  # hybrid
            bondstyle_name = line[1]
            style = self._get_style_by_name(BondStyle, bondstyle_name)
            if style is None:
                style = self.forcefield.def_bondstyle(bondstyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        # Parse atom types from bond type name (e.g., "CT-CT" -> ["CT", "CT"])
        atom_names = self._parse_type_name(bond_type_name)
        if len(atom_names) >= 2:
            itom = self._ensure_atomtype(atom_names[0])
            jtom = self._ensure_atomtype(atom_names[1])
        else:
            # Fallback: use same atom type for both
            itom = jtom = self._ensure_atomtype(atom_names[0])

        # Convert coeffs to kwargs based on style
        # For harmonic bond: k, r0
        kwargs = {}
        if len(coeffs) >= 2:
            kwargs["k"] = float(coeffs[0])
            kwargs["r0"] = float(coeffs[1])

        style.def_type(itom, jtom, name=bond_type_name, **kwargs)

    def read_angle_coeff(self, style, line):
        """Read angle_coeff line and create AngleType.

        Format: angle_coeff <type_name> [style_name] <k> <theta0>
        Example: angle_coeff CT-CT-OS 50.0 109.5
        """
        angle_type_name = line[0]

        if line[1].isalpha():  # hybrid
            anglestyle_name = line[1]
            style = self._get_style_by_name(AngleStyle, anglestyle_name)
            if style is None:
                style = self.forcefield.def_anglestyle(anglestyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        # Parse atom types from angle type name (e.g., "CT-CT-OS" -> ["CT", "CT", "OS"])
        atom_names = self._parse_type_name(angle_type_name)
        if len(atom_names) >= 3:
            itom = self._ensure_atomtype(atom_names[0])
            jtom = self._ensure_atomtype(atom_names[1])
            ktom = self._ensure_atomtype(atom_names[2])
        else:
            # Fallback: use same atom type for all
            itom = jtom = ktom = self._ensure_atomtype(atom_names[0])

        # Convert coeffs to kwargs based on style
        # For harmonic angle: k, theta0
        kwargs = {}
        if len(coeffs) >= 2:
            kwargs["k"] = float(coeffs[0])
            kwargs["theta0"] = float(coeffs[1])

        style.def_type(itom, jtom, ktom, name=angle_type_name, **kwargs)

    def read_dihedral_coeff(self, style, line):
        """Read dihedral_coeff line and create DihedralType.

        Format: dihedral_coeff <type_name> [style_name] <k1> <k2> <k3> <k4>
        Example: dihedral_coeff CT-CT-CT-CT 1.3 -0.05 0.2 0.0
        """
        dihedral_type_name = line[0]

        if (
            not line[1].isdigit() and "." not in line[1] and line[1] != "-"
        ):  # hybrid (not a number)
            dihedralsyle_name = line[1]
            style = self._get_style_by_name(DihedralStyle, dihedralsyle_name)
            if style is None:
                style = self.forcefield.def_dihedralstyle(dihedralsyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        # Parse atom types from dihedral type name
        atom_names = self._parse_type_name(dihedral_type_name)
        if len(atom_names) >= 4:
            itom = self._ensure_atomtype(atom_names[0])
            jtom = self._ensure_atomtype(atom_names[1])
            ktom = self._ensure_atomtype(atom_names[2])
            ltom = self._ensure_atomtype(atom_names[3])
        else:
            # Fallback: use same atom type for all
            itom = jtom = ktom = ltom = self._ensure_atomtype(atom_names[0])

        # Convert coeffs to kwargs based on style
        # For OPLS dihedral: k1, k2, k3, k4 (LAMMPS format)
        # Note: LAMMPS uses k1-k4, but XML uses c0-c5
        # Here we read LAMMPS format and store as c1-c4
        kwargs = {}
        if len(coeffs) >= 4:
            kwargs["c1"] = float(coeffs[0])  # k1 -> c1
            kwargs["c2"] = float(coeffs[1])  # k2 -> c2
            kwargs["c3"] = float(coeffs[2])  # k3 -> c3
            kwargs["c4"] = float(coeffs[3])  # k4 -> c4
            # Add c0 and c5 as 0.0 for compatibility with XML format
            kwargs["c0"] = 0.0
            kwargs["c5"] = 0.0

        style.def_type(itom, jtom, ktom, ltom, name=dihedral_type_name, **kwargs)

    def read_improper_coeff(self, style, line):
        """Read improper_coeff line and create ImproperType.

        Format: improper_coeff <type_name> [style_name] <params...>
        Example: improper_coeff CA-CA-CA-HA 1.1 180.0
        """
        improper_type_name = line[0]

        if line[1].isalpha():  # hybrid
            improperstyle_name = line[1]
            style = self._get_style_by_name(ImproperStyle, improperstyle_name)
            if style is None:
                style = self.forcefield.def_improperstyle(improperstyle_name)
            coeffs = line[2:]
        else:
            coeffs = line[1:]

        # Parse atom types from improper type name
        atom_names = self._parse_type_name(improper_type_name)
        if len(atom_names) >= 4:
            itom = self._ensure_atomtype(atom_names[0])
            jtom = self._ensure_atomtype(atom_names[1])
            ktom = self._ensure_atomtype(atom_names[2])
            ltom = self._ensure_atomtype(atom_names[3])
        else:
            # Fallback: use same atom type for all
            itom = jtom = ktom = ltom = self._ensure_atomtype(atom_names[0])

        # Convert coeffs to kwargs
        # For improper: k, chi0 (typical cvff/harmonic improper)
        kwargs = {}
        if len(coeffs) >= 2:
            kwargs["k"] = float(coeffs[0])
            kwargs["chi0"] = float(coeffs[1])

        style.def_type(itom, jtom, ktom, ltom, name=improper_type_name, **kwargs)

    def read_pair_coeff(self, style, line):
        """Read pair_coeff line and create PairType.

        Format can be:
        - pair_coeff <type> <epsilon> <sigma> (self-interaction)
        - pair_coeff <i> <j> <epsilon> <sigma> (cross-interaction)
        - pair_coeff <i> <j> [style_name] <epsilon> <sigma> (hybrid)

        Example: pair_coeff opls_135 0.066 3.5
        """
        # Determine format by checking if second element is a number
        if len(line) >= 3 and (
            line[1].replace(".", "").replace("-", "").isdigit()
            or line[1].replace(".", "").replace("e", "").replace("-", "").isdigit()
        ):
            # Format: pair_coeff <type> <epsilon> <sigma>
            i = j = line[0]
            coeffs = line[1:]
        elif len(line) >= 4:
            # Format: pair_coeff <i> <j> ...
            i, j = line[0], line[1]
            if len(line) > 4 and line[2].isalpha():  # hybrid (style name present)
                pairstyle_name = line[2]
                style = self._get_style_by_name(PairStyle, pairstyle_name)
                if style is None:
                    style = self.forcefield.def_pairstyle(pairstyle_name)
                coeffs = line[3:]
            else:
                coeffs = line[2:]
        else:
            # Fallback
            i = j = line[0]
            coeffs = line[1:] if len(line) > 1 else []

        # Ensure atom types exist
        atomtype_i = self._ensure_atomtype(i)
        atomtype_j = self._ensure_atomtype(j)

        # Convert coeffs to kwargs
        # For LJ pair: epsilon, sigma
        kwargs = {}
        if len(coeffs) >= 2:
            kwargs["epsilon"] = float(coeffs[0])
            kwargs["sigma"] = float(coeffs[1])

        # Generate name for the pair type
        if atomtype_i == atomtype_j:
            pair_name = atomtype_i.name
        else:
            pair_name = f"{atomtype_i.name}-{atomtype_j.name}"

        style.def_type(atomtype_i, atomtype_j, name=pair_name, **kwargs)

    def read_pair_modify(self, line):
        if line[0] == "pair":
            raise NotImplementedError("pair_modify hybrid not implemented")
        else:
            pairstyles = self.forcefield.get_styles(PairStyle)
            assert len(pairstyles) == 1, ValueError(
                "pair_modify command requires one pair style"
            )
            # Exactly one pair style exists (asserted above). molrs styles are
            # immutable handles that cannot store a string-list param, so record
            # the pair_modify args on the reader instead of on the style.
            modified = getattr(self, "_pair_modify", None)
            if modified is None:
                self._pair_modify = list(line)
            else:
                for token in line:
                    if token not in modified:
                        modified.append(token)


# ===================================================================
#               Parameter Formatters (CL&Pol / specialized)
# ===================================================================
#
# Full ``*.ff`` write for the AMBER/GAFF flavour lives in molrs
# (``molrs.write_lammps_forcefield``): unit conversion is the inverse of
# ``LammpsFfReader`` and must not be reimplemented here. These formatters remain
# for specialized pair styles (CL&Pol Thole / Tang−Toennies) that the native
# writer does not yet emit — they feed ``LammpsForceFieldFormatter.format_params``
# only, not the main include path.


def _format_pair_thole(typ) -> list[float]:
    """Format a Thole pair type's parameters: alpha a_thole.

    CL&Pol Thole core–shell screening (LAMMPS ``pair_style thole``).
    """
    kwargs = typ.params.kwargs
    return [kwargs.get("alpha", 0.0), kwargs.get("a_thole", 2.6)]


def _format_pair_coul_tt(typ) -> list[float]:
    """Format a Tang−Toennies pair type's parameters: b n c.

    CL&Pol Tang−Toennies charge–dipole damping (LAMMPS ``pair_style coul/tt``).
    """
    kwargs = typ.params.kwargs
    return [kwargs.get("b", 4.5), kwargs.get("n", 4), kwargs.get("c", 1.0)]


class LammpsForceFieldFormatter(LammpsFieldFormatter, ForceFieldFormatter):
    """LAMMPS force-field parameter formatter for specialized styles.

    AMBER/GAFF bond/angle/dihedral/pair serialization is owned by molrs
    (``write_lammps_forcefield``). This registry keeps CL&Pol damping pair
    formatters until those styles also sink into the native writer.
    """

    _param_formatters = {
        PairTholeStyle: _format_pair_thole,
        PairCoulTTStyle: _format_pair_coul_tt,
    }


# ===================================================================
#               LAMMPS Force Field Writer
# ===================================================================


class LAMMPSForceFieldWriter:
    """Writer for LAMMPS force-field includes (``*.ff``).

    Thin shell over :func:`molrs.write_lammps_forcefield` /
    :func:`molrs.write_lammps_forcefield_str`. Unit conversion (``K = k/2``,
    angles radians → degrees, fourier phase, combined ``lj/cut/coul/cut``) lives
    in molrs as the inverse of the native reader — molpy does not reimplement it.
    """

    def __init__(self, fpath: str | Path | TextIO, precision: int = 6):
        """
        Args:
            fpath: Output file path or file-like object.
            precision: Decimal places for floating-point coefficients.
        """
        self.precision = precision
        self._fpath = fpath

    def write(
        self,
        forcefield: ForceField,
        atom_types: set[str] | None = None,
        bond_types: set[str] | None = None,
        angle_types: set[str] | None = None,
        dihedral_types: set[str] | None = None,
        improper_types: set[str] | None = None,
        skip_pair_style: bool = False,
    ) -> None:
        """Write forcefield to LAMMPS format via molrs.

        Args:
            forcefield: ForceField in molrs units.
            atom_types: Optional atom-type whitelist for pair coeffs.
            bond_types: Optional bond type-name whitelist.
            angle_types: Optional angle type-name whitelist.
            dihedral_types: Optional dihedral type-name whitelist.
            improper_types: Optional improper type-name whitelist.
            skip_pair_style: If True, omit the ``pair_style`` header line.
        """
        import molrs

        kwargs = dict(
            precision=self.precision,
            skip_pair_style=skip_pair_style,
            atom_types=atom_types,
            bond_types=bond_types,
            angle_types=angle_types,
            dihedral_types=dihedral_types,
            improper_types=improper_types,
        )
        if isinstance(self._fpath, (str, Path)):
            molrs.write_lammps_forcefield(str(self._fpath), forcefield, **kwargs)
        else:
            self._fpath.write(molrs.write_lammps_forcefield_str(forcefield, **kwargs))
