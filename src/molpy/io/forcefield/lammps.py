from itertools import islice
from pathlib import Path
from typing import Callable, TextIO, cast

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
    Style,
)
from molpy.potential.angle import AngleHarmonicStyle
from molpy.potential.bond import BondHarmonicStyle
from molpy.potential.dihedral import DihedralOPLSStyle
from molpy.potential.pair import PairLJ126CoulCutStyle, PairLJ126CoulLongStyle
from molpy.version import version


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
        for type_class in atomstyle.types.classes():
            for atom_type in atomstyle.types.bucket(type_class):
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
            pairstyle = pairstyles[0]

            if "modified" in pairstyle.params.kwargs:
                for l in line:
                    if l not in pairstyle.params.kwargs["modified"]:
                        pairstyle.params.kwargs["modified"].append(l)
            else:
                pairstyle.params.kwargs["modified"] = line


# ===================================================================
#               Type Filter
# ===================================================================


class TypeFilter:
    """Filter for selecting which types to include in LAMMPS output.

    Supports multiple filtering modes:
    - whitelist: Only include types whose names are in the set
    - blacklist: Exclude types whose names are in the set
    - custom: Use a custom function to determine inclusion
    """

    def __init__(
        self,
        whitelist: set[str] | None = None,
        blacklist: set[str] | None = None,
        custom: Callable | None = None,
    ):
        """
        Args:
            whitelist: Set of type names to include. If None, all types pass.
            blacklist: Set of type names to exclude. Applied after whitelist.
            custom: Custom filter function that takes (type_obj) -> bool.
                   Applied after whitelist and blacklist.
        """
        self.whitelist = whitelist
        self.blacklist = blacklist or set()
        self.custom = custom

    def includes(self, type_obj) -> bool:
        """Check if a type should be included.

        Args:
            type_obj: Type object to check

        Returns:
            True if type should be included, False otherwise
        """
        # Apply whitelist
        if self.whitelist is not None:
            if type_obj.name not in self.whitelist:
                return False

        # Apply blacklist
        if type_obj.name in self.blacklist:
            return False

        # Apply custom filter
        if self.custom is not None:
            if not self.custom(type_obj):
                return False

        return True

    @classmethod
    def from_whitelist(cls, whitelist: set[str] | None) -> "TypeFilter":
        """Create a filter from a whitelist (backward compatibility)."""
        return cls(whitelist=whitelist)


# ===================================================================
#               Parameter Formatters
# ===================================================================


def _format_bond_harmonic(typ) -> list[float]:
    """Format BondHarmonicType parameters: k r0"""
    from molpy.potential.bond import BondHarmonicType

    if isinstance(typ, BondHarmonicType):
        return [typ.params.kwargs["k"], typ.params.kwargs["r0"]]
    # Fallback for generic BondType
    return [typ.params.kwargs["k"], typ.params.kwargs["r0"]]


def _format_angle_harmonic(typ) -> list[float]:
    """Format AngleHarmonicType parameters: k theta0

    Parameters are already in LAMMPS format (kcal/mol/rad² for k, degrees for theta0),
    so no conversion is needed - just return them directly.
    """
    from molpy.potential.angle import AngleHarmonicType

    k = typ.params.kwargs.get("k", 0.0)  # kcal/mol/rad² (already in LAMMPS format)
    theta0 = typ.params.kwargs.get("theta0", 0.0)  # degrees

    return [k, theta0]


def _format_dihedral_opls(typ) -> list[float]:
    """Format DihedralOPLSType parameters: k1 k2 k3 k4

    Uses analytical RB → OPLS conversion via rb_to_opls() function.
    The c0-c5 coefficients are converted to LAMMPS k1-k4 format.
    """
    from molpy.potential.dihedral import DihedralOPLSType

    if isinstance(typ, DihedralOPLSType):
        return [
            typ.params.kwargs.get("c1", 0.0),
            typ.params.kwargs.get("c2", 0.0),
            typ.params.kwargs.get("c3", 0.0),
            typ.params.kwargs.get("c4", 0.0),
        ]
    # Fallback for generic DihedralType
    kwargs = typ.params.kwargs
    return [
        kwargs.get("c1", 0.0),
        kwargs.get("c2", 0.0),
        kwargs.get("c3", 0.0),
        kwargs.get("c4", 0.0),
    ]


def _format_pair_lj(typ) -> list[float]:
    """Format PairLJ126Type parameters: epsilon sigma"""
    from molpy.potential.pair import PairLJ126Type

    if isinstance(typ, PairLJ126Type):
        result = []
        if "epsilon" in typ.params.kwargs:
            result.append(typ.params.kwargs["epsilon"])
        if "sigma" in typ.params.kwargs:
            result.append(typ.params.kwargs["sigma"])
        return result
    # Fallback for generic PairType
    kwargs = typ.params.kwargs
    result = []
    if "epsilon" in kwargs:
        result.append(kwargs["epsilon"])
    if "sigma" in kwargs:
        result.append(kwargs["sigma"])
    return result


def _format_generic_bond(typ) -> list[float]:
    """Format generic BondType parameters: k r0"""
    kwargs = typ.params.kwargs
    return [kwargs.get("k", 0.0), kwargs.get("r0", 0.0)]


def _format_generic_angle(typ) -> list[float]:
    """Format generic AngleType parameters: k theta0"""
    kwargs = typ.params.kwargs
    return [kwargs.get("k", 0.0), kwargs.get("theta0", 0.0)]


def _format_generic_dihedral(typ) -> list[float]:
    """Format generic DihedralType parameters: k1 k2 k3 k4"""
    kwargs = typ.params.kwargs
    return [
        kwargs.get("k1", kwargs.get("c1", 0.0)),
        kwargs.get("k2", kwargs.get("c2", 0.0)),
        kwargs.get("k3", kwargs.get("c3", 0.0)),
        kwargs.get("k4", kwargs.get("c4", 0.0)),
    ]


def _format_generic_pair(typ) -> list[float]:
    """Format generic PairType parameters: epsilon sigma"""
    kwargs = typ.params.kwargs
    result = []
    if "epsilon" in kwargs:
        result.append(kwargs["epsilon"])
    if "sigma" in kwargs:
        result.append(kwargs["sigma"])
    return result


# Parameter formatters registry: maps Style class to formatter function
_PARAM_FORMATTERS: dict[type, callable] = {
    # Specialized styles
    BondHarmonicStyle: _format_bond_harmonic,
    AngleHarmonicStyle: _format_angle_harmonic,
    DihedralOPLSStyle: _format_dihedral_opls,
    PairLJ126CoulCutStyle: _format_pair_lj,
    PairLJ126CoulLongStyle: _format_pair_lj,
    # Generic styles (fallback)
    BondStyle: _format_generic_bond,
    AngleStyle: _format_generic_angle,
    DihedralStyle: _format_generic_dihedral,
    PairStyle: _format_generic_pair,
}


# ===================================================================
#               LAMMPS Force Field Writer
# ===================================================================


class LAMMPSForceFieldWriter:
    """Writer for LAMMPS force field files.

    Converts ForceField objects to LAMMPS input format with support for:
    - Multiple style types (bond, angle, dihedral, improper, pair)
    - Hybrid styles
    - Type filtering
    - Specialized Style and Type classes
    """

    def __init__(self, fpath: str | Path | TextIO, precision: int = 6):
        """
        Args:
            fpath: Output file path or file-like object
            precision: Number of decimal places for floating point values
        """
        self.precision = precision
        self._fpath = fpath

    def _format_number(self, value: float | int) -> str:
        """Format a single number with configured precision."""
        if isinstance(value, float):
            return f"{value:.{self.precision}f}"
        return str(value)

    def _format_params(self, params: list[float | int]) -> str:
        """Format a list of parameters for LAMMPS output."""
        return " ".join(self._format_number(p) for p in params)

    def _get_type_params(self, typ, style) -> list[float]:
        """Extract parameters from a Type object for LAMMPS coefficients.

        Args:
            typ: Type object (BondType, AngleType, etc.)
            style: Style object that contains this type

        Returns:
            List of parameters in LAMMPS format

        Raises:
            ValueError: If no formatter is found and parameters cannot be extracted
        """
        style_class = type(style)

        # Try registered formatter first
        if style_class in _PARAM_FORMATTERS:
            formatter = _PARAM_FORMATTERS[style_class]
            try:
                return formatter(typ)
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"Failed to format parameters for {style_class.__name__} "
                    f"with type {type(typ).__name__}: {e}"
                ) from e

        # No formatter found - this is an error for specialized styles
        raise ValueError(
            f"No parameter formatter registered for style class {style_class.__name__}. "
            f"Available formatters: {list(_PARAM_FORMATTERS.keys())}"
        )

    def _get_coeff_id(self, typ, style_type: str) -> str:
        """Get coefficient identifier for a type.

        Args:
            typ: Type object
            style_type: Style type name ("bond", "angle", "pair", etc.)

        Returns:
            Coefficient identifier string for LAMMPS
        """
        if style_type == "pair":
            # Pair types need "I J" format
            if hasattr(typ, "itom") and hasattr(typ, "jtom"):
                return f"{typ.itom.name} {typ.jtom.name}"
            raise ValueError(f"PairType {typ} missing itom/jtom attributes")
        else:
            # Other types use name directly
            return typ.name

    def _get_style_params(self, style) -> list[float]:
        """Get style parameters (cutoffs, etc.) from style.params.args."""
        return list(style.params.args) if style.params.args else []

    def _get_default_style_params(
        self, style_name: str, style_type: str
    ) -> list[float]:
        """Get default parameters for a style if none are specified."""
        if style_type == "pair":
            if style_name in ["lj/cut/coul/cut", "lj/cut/coul/long"]:
                return [10.0, 10.0]  # LJ cutoff, Coulomb cutoff
            elif style_name in ["lj/cut", "lj126"]:
                return [10.0]  # Single cutoff
        return []

    def _write_style_header(
        self,
        lines: list[str],
        style: Style,
        style_type: str,
    ) -> None:
        """Write style header line (e.g., 'bond_style harmonic').

        Args:
            lines: Output lines list
            style: Style object
            style_type: Style type name
        """
        params = self._get_style_params(style)
        if not params:
            params = self._get_default_style_params(style.name, style_type)

        style_line = f"{style_type}_style {style.name}"
        if params:
            style_line += f" {self._format_params(params)}"
        lines.append(style_line + "\n")

    def _write_style_modify(
        self,
        lines: list[str],
        style: Style,
        style_type: str,
    ) -> None:
        """Write style modify line if present.

        Args:
            lines: Output lines list
            style: Style object
            style_type: Style type name
        """
        if "modified" in style.params.kwargs:
            modify_args = " ".join(style.params.kwargs["modified"])
            lines.append(f"{style_type}_modify {modify_args}\n")

    def _write_type_coeffs(
        self,
        lines: list[str],
        style: Style,
        style_type: str,
        type_filter: TypeFilter,
    ) -> None:
        """Write coefficient lines for all types in a style.

        Args:
            lines: Output lines list
            style: Style object
            style_type: Style type name
            type_filter: Filter to determine which types to include
        """
        has_types = False
        for type_class in style.types.classes():
            types = [
                t for t in style.types.bucket(type_class) if type_filter.includes(t)
            ]
            if types:
                has_types = True
                for typ in types:
                    params = self._get_type_params(typ, style)
                    coeff_id = self._get_coeff_id(typ, style_type)
                    lines.append(
                        f"{style_type}_coeff {coeff_id} {self._format_params(params)}\n"
                    )

        if has_types:
            lines.append("\n")

    def _write_single_style_section(
        self,
        lines: list[str],
        style: Style,
        style_type: str,
        type_filter: TypeFilter,
    ) -> None:
        """Write a section for a single style (non-hybrid).

        Args:
            lines: Output lines list
            style: Style object
            style_type: Style type name
            type_filter: Filter to determine which types to include
        """
        self._write_style_header(lines, style, style_type)
        self._write_style_modify(lines, style, style_type)
        self._write_type_coeffs(lines, style, style_type, type_filter)

    def _write_hybrid_style_section(
        self,
        lines: list[str],
        styles: list[Style],
        style_type: str,
        type_filter: TypeFilter,
    ) -> None:
        """Write a section for hybrid styles.

        Args:
            lines: Output lines list
            styles: List of Style objects
            style_type: Style type name
            type_filter: Filter to determine which types to include
        """
        style_names = " ".join(s.name for s in styles)
        lines.append(f"{style_type}_style hybrid {style_names}\n")
        lines.append("\n")

        for style in styles:
            for type_class in style.types.classes():
                types = [
                    t for t in style.types.bucket(type_class) if type_filter.includes(t)
                ]
                for typ in types:
                    params = self._get_type_params(typ, style)
                    coeff_id = self._get_coeff_id(typ, style_type)
                    lines.append(
                        f"{style_type}_coeff {coeff_id} {style.name} {self._format_params(params)}\n"
                    )

        lines.append("\n")

    def _write_style_section(
        self,
        lines: list[str],
        styles: list[Style],
        style_type: str,
        type_filter: TypeFilter,
    ) -> None:
        """Write a complete style section.

        Args:
            lines: Output lines list
            styles: List of Style objects
            style_type: Style type name
            type_filter: Filter to determine which types to include
        """
        if not styles:
            return

        if len(styles) == 1:
            self._write_single_style_section(lines, styles[0], style_type, type_filter)
        else:
            self._write_hybrid_style_section(lines, styles, style_type, type_filter)

    def write(
        self,
        forcefield: ForceField,
        atom_types: set[str] | None = None,
        bond_types: set[str] | None = None,
        angle_types: set[str] | None = None,
        dihedral_types: set[str] | None = None,
        improper_types: set[str] | None = None,
    ) -> None:
        """Write forcefield to LAMMPS format.

        Args:
            forcefield: ForceField object to write
            atom_types: Set of atom type names to include (for pair coeffs).
                       If None, include all.
            bond_types: Set of bond type names to include. If None, include all.
            angle_types: Set of angle type names to include. If None, include all.
            dihedral_types: Set of dihedral type names to include. If None, include all.
            improper_types: Set of improper type names to include. If None, include all.
        """
        lines = [f"# LAMMPS force field generated by molpy version {version}\n\n"]

        # Create type filters
        filters = {
            "pair": TypeFilter.from_whitelist(atom_types),
            "bond": TypeFilter.from_whitelist(bond_types),
            "angle": TypeFilter.from_whitelist(angle_types),
            "dihedral": TypeFilter.from_whitelist(dihedral_types),
            "improper": TypeFilter.from_whitelist(improper_types),
        }

        # Get styles and write sections
        style_configs = [
            (forcefield.get_styles(PairStyle), "pair"),
            (forcefield.get_styles(BondStyle), "bond"),
            (forcefield.get_styles(AngleStyle), "angle"),
            (forcefield.get_styles(DihedralStyle), "dihedral"),
            (forcefield.get_styles(ImproperStyle), "improper"),
        ]

        for styles, style_type in style_configs:
            self._write_style_section(lines, styles, style_type, filters[style_type])

        # Write to file
        if isinstance(self._fpath, (str, Path)):
            with open(self._fpath, "w") as f:
                f.writelines(lines)
        else:
            self._fpath.writelines(lines)
