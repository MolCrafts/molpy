"""XML force field parser for atomistic force fields."""

import logging
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from molpy.core.forcefield import AtomisticForcefield, AtomType
from molpy.potential.angle import AngleHarmonicStyle
from molpy.potential.bond import BondHarmonicStyle
from molpy.potential.dihedral import DihedralOPLSStyle
from molpy.potential.pair import PairLJ126CoulCutStyle, PairLJ126CoulLongStyle

logger = logging.getLogger(__name__)


def _normalize_to_wildcard(value: str | None) -> str:
    """
    Normalize None or empty string to wildcard "*".

    Args:
        value: Input value (can be None, "", or any string)

    Returns:
        "*" if value is None or "", otherwise the original value
    """
    if value is None or value == "":
        return "*"
    return value


def _get_canonical_class(class_: str, aliases: dict[str, str]) -> str:
    """
    Get canonical form of class (resolve aliases)

    Example: CT_2 -> CT, CT_3 -> CT

    Args:
        class_: Original class name
        aliases: Alias mapping dictionary

    Returns:
        Canonical class name
    """
    # If wildcard or not in alias table, return directly
    if class_ == "*" or class_ not in aliases:
        return class_
    # Recursive resolution (prevent multi-level aliases)
    canonical = aliases[class_]
    if canonical in aliases:
        return _get_canonical_class(canonical, aliases)
    return canonical


def _resolve_forcefield_path(filepath: str | Path) -> Path:
    """
    Resolve force field file path, checking built-in data directory first.

    Args:
        filepath: Path to force field file, or just filename for built-in files

    Returns:
        Resolved Path object

    Raises:
        FileNotFoundError: If file not found in any location
    """
    filepath = Path(filepath)

    # If it's just a filename (e.g., "oplsaa.xml"), check built-in data
    if filepath.name == str(filepath):
        # Try built-in data directory using the new data module
        try:
            from molpy.data import get_forcefield_path

            builtin_path = get_forcefield_path(filepath.name)
            logger.info(f"Using built-in force field: {builtin_path}")
            return Path(builtin_path)
        except FileNotFoundError:
            # File not found in built-in data, continue to check provided path
            pass

    # Otherwise use the provided path
    if filepath.exists():
        return filepath

    raise FileNotFoundError(
        f"Force field file not found: {filepath}. "
        f"Available built-in force fields: {_list_available_forcefields()}"
    )


def _list_available_forcefields() -> list[str]:
    """List available built-in force fields."""
    try:
        from molpy.data import list_forcefields

        return list_forcefields()
    except Exception:
        return []


class XMLForceFieldReader:
    """
    XML force field parser for atomistic force fields.

    Parses XML-formatted force field files (e.g., OPLS-AA) and populates
    an AtomisticForcefield object with atom types, bond parameters, angle
    parameters, dihedral parameters, and nonbonded interactions.

    The parser handles:
    - AtomTypes section: atom type definitions
    - HarmonicBondForce: harmonic bond parameters
    - HarmonicAngleForce: harmonic angle parameters
    - RBTorsionForce: Ryckaert-Bellemans dihedral parameters
    - NonbondedForce: LJ and Coulomb parameters
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize the XML force field reader.

        Args:
            filepath: Path to the XML force field file, or filename for built-in files
                     (e.g., "oplsaa.xml" will load from molpy/data/forcefield/)
        """
        self._file = _resolve_forcefield_path(filepath)
        self._type_to_atomtype: dict[str, AtomType] = {}  # type -> AtomType mapping
        self._class_to_atomtype: dict[str, AtomType] = {}  # class -> AtomType mapping
        self._any_atomtype: AtomType | None = None  # Full wildcard ("*", "*")
        # Class alias mapping: derived class -> canonical base class
        # Example: CT_2 -> CT, CT_3 -> CT
        self._class_aliases: dict[str, str] = {}
        # Defined class set: classes explicitly defined in AtomTypes
        # These classes should not be automatically resolved as aliases
        self._defined_classes: set[str] = set()
        # overrides mapping: type -> overridden_type
        # Example: opls_961 overrides opls_962
        self._overrides: dict[str, str] = {}

        self._ff: AtomisticForcefield | None = None

    def read(
        self, forcefield: AtomisticForcefield | None = None
    ) -> AtomisticForcefield:
        """
        Read and parse the XML force field file.

        Args:
            forcefield: Optional existing force field to populate. If None, creates new one.

        Returns:
            Populated AtomisticForcefield object
        """
        if not self._file.exists():
            raise FileNotFoundError(f"Force field file not found: {self._file}")

        tree = ET.parse(self._file)
        root = tree.getroot()

        # Get force field metadata from root element
        ff_name = root.get("name", "Unknown")
        ff_version = root.get("version", "0.0.0")
        combining_rule = root.get("combining_rule", "geometric")

        # Create or use provided force field
        if forcefield is None:
            self._ff = AtomisticForcefield(name=ff_name, units="real")
        else:
            self._ff = forcefield

        logger.info(f"Parsing force field: {ff_name} v{ff_version}")
        logger.info(f"Combining rule: {combining_rule}")

        # Parse all sections
        for child in root:
            tag = child.tag
            if tag == "AtomTypes":
                self._parse_atomtypes(child)
            elif tag == "HarmonicBondForce":
                self._parse_bonds(child)
            elif tag == "HarmonicAngleForce":
                self._parse_angles(child)
            elif tag == "RBTorsionForce":
                self._parse_dihedrals(child)
            elif tag == "NonbondedForce":
                self._parse_nonbonded(child)
            else:
                logger.debug(f"Skipping unknown section: {tag}")

        logger.info(f"Parsed {len(self._type_to_atomtype)} atom types (by type)")
        return self._ff

    def _resolve_atomtype_with_alias(self, class_str: str) -> AtomType:
        """
        Parse class string, if not exist try to establish alias mapping.

        Strategy: If class_str is like "XX_N" (N is digit), and XX exists in class_to_atomtype,
        establish XX_N -> XX alias mapping and return AtomType for XX.

        Args:
            class_str: Class string (from class attribute of BondType/AngleType/DihedralType)

        Returns:
            Corresponding AtomType

        Raises:
            KeyError: If Corresponding AtomType not found
        """
        # Try direct lookup first
        if class_str in self._class_to_atomtype:
            return self._class_to_atomtype[class_str]

        # If not found, check if it's derived class (XX_N form)
        if "_" in class_str:
            parts = class_str.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_class = parts[0]
                # Check if base class exists
                if base_class in self._class_to_atomtype:
                    # Establish alias mapping
                    self._class_aliases[class_str] = base_class
                    logger.debug(
                        f"Auto-registered class alias: {class_str} -> {base_class}"
                    )
                    return self._class_to_atomtype[base_class]

        # If neither found, raise exception
        raise KeyError(f"AtomType with class '{class_str}' not found")

    def _get_or_create_atomtype(
        self, type_: str, class_: str, **kwargs: Any
    ) -> AtomType:
        """
        Get or create an AtomType with exact type_ and class_.

        Use alias mapping: derived classes (like CT_2, CT_3) normalized to base class (CT)

        Maintain three mapping tables:
        1. type_to_atomtype: type -> AtomType (only when type != "*")
        2. class_to_atomtype: class -> AtomType (only when class != "*")
        3. any_atomtype: Full wildcard ("*", "*")

        Args:
            type_: Type identifier (normalized, None or "" converted to "*")
            class_: Class identifier (normalized, None or "" converted to "*")
            **kwargs: Other parameters (element, mass, etc.)

        Returns:
            AtomType instance
        """
        # Normalize input
        type_ = _normalize_to_wildcard(type_)
        class_ = _normalize_to_wildcard(class_)

        # Normalize class (resolve aliases)
        # If class not in _class_to_atomtype, try alias resolution
        canonical_class = class_
        if class_ != "*" and class_ not in self._class_to_atomtype:
            # Try alias resolution
            if class_ in self._class_aliases:
                canonical_class = _get_canonical_class(class_, self._class_aliases)
            elif "_" in class_:
                # Auto-detect and register aliases (XX_N -> XX)
                # However, if this class is already defined in AtomTypes, should not resolve it as alias
                # Example: O_3 is an independent class, should not be resolved as an alias of O
                if class_ not in self._defined_classes:
                    parts = class_.rsplit("_", 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        base_class = parts[0]
                        if base_class in self._class_to_atomtype:
                            self._class_aliases[class_] = base_class
                            canonical_class = base_class
                            logger.debug(
                                f"Auto-registered class alias: {class_} -> {base_class}"
                            )
        elif class_ in self._class_aliases:
            canonical_class = _get_canonical_class(class_, self._class_aliases)

        # Determine name: type first, else canonical_class, else "*" "*"
        if type_ != "*":
            name = type_
        elif canonical_class != "*":
            name = canonical_class
        else:
            name = "*"

        # Add original type_ and canonical class_ to kwargs
        kwargs["type_"] = type_
        kwargs["class_"] = canonical_class  # Use normalized class
        # also expose a full_name (type-class) to preserve both identifiers
        if type_ != "*":
            kwargs.setdefault("full_name", f"{type_}-{canonical_class}")
        else:
            kwargs.setdefault("full_name", canonical_class)

        # Case 1: Full wildcard ("*", "*")
        if type_ == "*" and canonical_class == "*":
            if self._any_atomtype is None:
                assert (
                    self._ff is not None
                ), "Force field must be initialized before creating atom types"
                atomstyle = self._ff.def_atomstyle("full")
                self._any_atomtype = atomstyle.def_type(name=name, **kwargs)
                # record overrides if present for a specific type (not wildcard)
                ov = kwargs.get("overrides")
                if ov and type_ != "*":
                    self._overrides[type_] = ov
                logger.debug("Created global wildcard atom type (*,*)")
            return self._any_atomtype

        # Case 2: Specific type, any class (type, "*")
        if type_ != "*" and canonical_class == "*":
            if type_ in self._type_to_atomtype:
                return self._type_to_atomtype[type_]
            assert (
                self._ff is not None
            ), "Force field must be initialized before creating atom types"
            atomstyle = self._ff.def_atomstyle("full")
            atomtype = atomstyle.def_type(name=name, **kwargs)
            ov = kwargs.get("overrides")
            if ov and type_ != "*":
                self._overrides[type_] = ov
            self._type_to_atomtype[type_] = atomtype
            logger.debug(f"Created atom type ({type_}, *)")
            return atomtype

        # Case 3: Any type, specific class ("*", class)
        if type_ == "*" and canonical_class != "*":
            if canonical_class in self._class_to_atomtype:
                return self._class_to_atomtype[canonical_class]
            assert (
                self._ff is not None
            ), "Force field must be initialized before creating atom types"
            atomstyle = self._ff.def_atomstyle("full")
            atomtype = atomstyle.def_type(name=name, **kwargs)
            ov = kwargs.get("overrides")
            if ov and type_ != "*":
                self._overrides[type_] = ov
            self._class_to_atomtype[canonical_class] = atomtype
            logger.debug(f"Created atom type (*, {class_})")
            return atomtype

        # Case 4: Specific type and class (type, class)
        # Prefer type mapping lookup
        if type_ in self._type_to_atomtype:
            return self._type_to_atomtype[type_]
        # Create new AtomType
        assert (
            self._ff is not None
        ), "Force field must be initialized before creating atom types"
        atomstyle = self._ff.def_atomstyle("full")
        atomtype = atomstyle.def_type(name=name, **kwargs)
        ov = kwargs.get("overrides")
        if ov and type_ != "*":
            self._overrides[type_] = ov
        # Store only in type_to_atomtype, not in class_to_atomtype
        # Because class_to_atomtype is only for storing (*, class) form generic AtomType
        self._type_to_atomtype[type_] = atomtype
        logger.debug(f"Created atom type ({type_}, {canonical_class})")
        return atomtype

    def _parse_atomtypes(self, element: ET.Element) -> None:
        """
        Parse AtomTypes section.

        Args:
            element: AtomTypes XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        self._ff.def_atomstyle("full")
        count = 0

        for type_elem in element:
            if type_elem.tag != "Type":
                continue

            # Extract attributes
            # In XML: "name" is the type, "class" is the class
            type_name = type_elem.get("name")  # May be None
            class_name = type_elem.get("class")  # May be None
            element_sym = type_elem.get("element")
            mass_str = type_elem.get("mass")
            def_str = type_elem.get("def")
            desc_str = type_elem.get("desc")
            doi_str = type_elem.get("doi")
            overrides_str = type_elem.get("overrides")

            # Parse mass
            mass = float(mass_str) if mass_str else 0.0

            # If class is defined, record in _defined_classes
            if class_name:
                self._defined_classes.add(class_name)

            # Create atom type using _get_or_create_atomtype
            # Use _normalize_to_wildcard to convert None to "*" "*"
            self._get_or_create_atomtype(
                type_=type_name or "",  # Empty string will be normalized to "*" "*"
                class_=class_name or "",
                element=element_sym,
                mass=mass,
                def_=def_str,
                desc=desc_str,
                doi=doi_str,
                overrides=overrides_str,
            )

            count += 1

        logger.info(f"Parsed {count} atom types")

    def _parse_bonds(self, element: ET.Element) -> None:
        """
        Parse HarmonicBondForce section.

        Args:
            element: HarmonicBondForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        bondstyle = self._ff.def_style(BondHarmonicStyle())
        count = 0

        for bond_elem in element:
            if bond_elem.tag != "Bond":
                continue

            # Get atom type identifiers
            class1 = bond_elem.get("class1", "*")
            class2 = bond_elem.get("class2", "*")
            type1 = bond_elem.get("type1", "*")
            type2 = bond_elem.get("type2", "*")

            # Get bond parameters
            length_str = bond_elem.get("length")
            k_str = bond_elem.get("k")

            # Check if types are present (None means missing, "" or "*" are valid wildcards)
            if type1 is None or type2 is None:
                logger.warning("Skipping bond without type information")
                continue

            # Get or create atom types (handles wildcards)
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)

            # Parse parameters
            r0 = float(length_str) if length_str else 0.0
            k = float(k_str) if k_str else 0.0

            # Define bond type
            bondstyle.def_type(at1, at2, k, r0)
            count += 1

        logger.info(f"Parsed {count} bond types")

    def _parse_angles(self, element: ET.Element) -> None:
        """
        Parse HarmonicAngleForce section.

        Args:
            element: HarmonicAngleForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        anglestyle = self._ff.def_style(AngleHarmonicStyle())
        count = 0

        for angle_elem in element:
            if angle_elem.tag != "Angle":
                continue

            # Get atom type identifiers
            class1 = angle_elem.get("class1", "*")
            class2 = angle_elem.get("class2", "*")
            class3 = angle_elem.get("class3", "*")
            type1 = angle_elem.get("type1", "*")
            type2 = angle_elem.get("type2", "*")
            type3 = angle_elem.get("type3", "*")

            # Get angle parameters
            angle_str = angle_elem.get("angle")
            k_str = angle_elem.get("k")

            # Check if types are present (None means missing, "" or "*" are valid wildcards)
            if type1 is None or type2 is None or type3 is None:
                logger.warning("Skipping angle without type information")
                continue

            # Get or create atom types (handles wildcards)
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)

            # Parse parameters
            theta0 = float(angle_str) if angle_str else 0.0  # Store in radians as in XML
            k = float(k_str) if k_str else 0.0

            # Define angle type (theta0 in radians)
            anglestyle.def_type(at1, at2, at3, k, theta0)
            count += 1

        logger.info(f"Parsed {count} angle types")

    def _parse_dihedrals(self, element: ET.Element) -> None:
        """
        Parse RBTorsionForce section (Ryckaert-Bellemans dihedrals).

        Args:
            element: RBTorsionForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        dihedralstyle = self._ff.def_style(DihedralOPLSStyle())
        count = 0

        for dihedral_elem in element:
            if dihedral_elem.tag != "Proper":
                continue

            # Get atom type identifiers
            class1 = dihedral_elem.get("class1", "*")
            class2 = dihedral_elem.get("class2", "*")
            class3 = dihedral_elem.get("class3", "*")
            class4 = dihedral_elem.get("class4", "*")
            type1 = dihedral_elem.get("type1", "*")
            type2 = dihedral_elem.get("type2", "*")
            type3 = dihedral_elem.get("type3", "*")
            type4 = dihedral_elem.get("type4", "*")

            # Check if types are present (None means missing, "" or "*" are valid wildcards)
            if type1 is None or type2 is None or type3 is None or type4 is None:
                logger.warning("Skipping dihedral without type information")
                continue

            # Get or create atom types (handles wildcards)
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)
            at4 = self._get_or_create_atomtype(type4, class4)

            # Parse RB coefficients (c0-c5)
            c0 = float(dihedral_elem.get("c0", "0.0"))
            c1 = float(dihedral_elem.get("c1", "0.0"))
            c2 = float(dihedral_elem.get("c2", "0.0"))
            c3 = float(dihedral_elem.get("c3", "0.0"))
            c4 = float(dihedral_elem.get("c4", "0.0"))
            c5 = float(dihedral_elem.get("c5", "0.0"))

            # Use standard API to define dihedral type
            dihedralstyle.def_type(at1, at2, at3, at4, c0, c1, c2, c3, c4, c5)
            count += 1

        logger.info(f"Parsed {count} dihedral types")

    def _parse_nonbonded(self, element: ET.Element) -> None:
        """
        Parse NonbondedForce section (LJ and Coulomb parameters).

        Args:
            element: NonbondedForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        # Get scaling factors
        coulomb14scale = element.get("coulomb14scale", "0.5")
        lj14scale = element.get("lj14scale", "0.5")

        pairstyle = self._ff.def_style(
            PairLJ126CoulCutStyle(
                coulomb14scale=float(coulomb14scale),
                lj14scale=float(lj14scale),
            )
        )
        count = 0

        for atom_elem in element:
            if atom_elem.tag != "Atom":
                continue

            # Get atom type identifier
            type_name = atom_elem.get("type")
            if not type_name:
                logger.warning("Skipping nonbonded atom without type")
                continue

            # Get or create atom type
            # In NonbondedForce, type usually corresponds to actual type
            # Try to find from existing atomtype first (may already defined in AtomTypes)
            atomtype: AtomType | None = None
            if type_name in self._type_to_atomtype:
                atomtype = self._type_to_atomtype[type_name]
            else:
                # If not found, create new (class is "*") "*"）
                atomtype = self._get_or_create_atomtype(type_=type_name, class_="*")

            # Parse parameters
            charge_str = atom_elem.get("charge")
            sigma_str = atom_elem.get("sigma")
            epsilon_str = atom_elem.get("epsilon")

            charge = float(charge_str) if charge_str else 0.0
            sigma = float(sigma_str) if sigma_str else 0.0
            epsilon = float(epsilon_str) if epsilon_str else 0.0

            # Define pair parameters (self-interaction)
            # If atomtype exists, update its params; else create new pair type
            pairstyle.def_type(atomtype, atomtype, epsilon, sigma, charge)
            count += 1

        logger.info(f"Parsed {count} nonbonded parameters")


def read_xml_forcefield(
    filepath: str | Path, forcefield: AtomisticForcefield | None = None
) -> AtomisticForcefield:
    """
    Convenience function to read an XML force field file.

    Args:
        filepath: Path to the XML force field file, or filename for built-in files
                 (e.g., "oplsaa.xml" will load from molpy/data/forcefield/)
        forcefield: Optional existing force field to populate

    Returns:
        Populated AtomisticForcefield object

    Example:
        >>> # Load built-in OPLS-AA force field
        >>> ff = read_xml_forcefield("oplsaa.xml")
        >>>
        >>> # Load custom force field from path
        >>> from pathlib import Path
        >>> ff = read_xml_forcefield(Path("/path/to/custom.xml"))
    """
    # Check if this is oplsaa.xml and use specialized reader
    filepath_obj = Path(filepath)
    if filepath_obj.name == "oplsaa.xml" or str(filepath).endswith("oplsaa.xml"):
        return read_oplsaa_forcefield(filepath, forcefield)

    reader = XMLForceFieldReader(filepath)
    return reader.read(forcefield)


class OPLSAAForceFieldReader(XMLForceFieldReader):
    """
    Specialized reader for OPLS-AA force field with LAMMPS unit conversions.

    This reader extends XMLForceFieldReader to:
    1. Use lj/cut/coul/long pair style instead of lj/cut/coul/cut
    2. Convert epsilon: kJ/mol → kcal/mol (÷4.184)
    3. Convert sigma: nm → Å (×10)
    4. Convert bond K: kJ/mol/nm² → kcal/mol/Å² (÷(4.184×100), both use 0.5 factor in formula)
    5. Convert angle K: kJ/mol/rad² → kcal/mol/deg² (direct to LAMMPS format)
       Formula: k_lammps = (0.5 * k_opls / 4.184) * (π/180)²
       Stored internally in LAMMPS format, so no conversion needed when writing
    6. Convert dihedral K: kJ/mol → kcal/mol (÷4.184)
    """

    def _parse_bonds(self, element: ET.Element) -> None:
        """
        Parse HarmonicBondForce section with OPLS-AA unit conversions.

        Args:
            element: HarmonicBondForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        bondstyle = self._ff.def_style(BondHarmonicStyle())
        count = 0

        for bond_elem in element:
            if bond_elem.tag != "Bond":
                continue

            # Get atom type identifiers
            class1 = bond_elem.get("class1", "*")
            class2 = bond_elem.get("class2", "*")
            type1 = bond_elem.get("type1", "*")
            type2 = bond_elem.get("type2", "*")

            # Get bond parameters
            length_str = bond_elem.get("length")
            k_str = bond_elem.get("k")

            # Check if types are present
            if type1 is None or type2 is None:
                logger.warning("Skipping bond without type information")
                continue

            # Get or create atom types
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)

            # Parse parameters and convert units
            r0_nm = float(length_str) if length_str else 0.0
            k_opls = float(k_str) if k_str else 0.0

            # Convert units for LAMMPS
            # OPLS XML: E = 0.5 * k_opls * (r - r0)^2 (kJ/mol, nm)
            # LAMMPS harmonic bond: E = 0.5 * k * (r - r0)^2 (kcal/mol, Å)
            # Both have 0.5 factor, so only unit conversion needed
            # k: kJ/mol/nm² -> kcal/mol/Å²: divide by (4.184 * 100)
            # But if LAMMPS bond lacks the 0.5 factor, need: k = 0.5 * k_opls / (4.184 * 100)
            r0 = r0_nm * 10.0  # nm to Angstrom
            k = (
                0.5 * k_opls / (4.184 * 100)
            )  # kJ/mol/nm² to kcal/mol/Å² (accounting for 0.5 factor difference)

            # Define bond type
            bondstyle.def_type(at1, at2, k, r0)
            count += 1

        logger.info(f"Parsed {count} bond types (OPLS-AA with unit conversion)")

    def _parse_angles(self, element: ET.Element) -> None:
        """
        Parse HarmonicAngleForce section with OPLS-AA unit conversions.

        Args:
            element: HarmonicAngleForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        anglestyle = self._ff.def_style(AngleHarmonicStyle())
        count = 0

        for angle_elem in element:
            if angle_elem.tag != "Angle":
                continue

            # Get atom type identifiers
            class1 = angle_elem.get("class1", "*")
            class2 = angle_elem.get("class2", "*")
            class3 = angle_elem.get("class3", "*")
            type1 = angle_elem.get("type1", "*")
            type2 = angle_elem.get("type2", "*")
            type3 = angle_elem.get("type3", "*")

            # Get angle parameters
            angle_str = angle_elem.get("angle")
            k_str = angle_elem.get("k")

            # Check if types are present
            if type1 is None or type2 is None or type3 is None:
                logger.warning("Skipping angle without type information")
                continue

            # Get or create atom types
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)

            # Parse parameters and convert units
            theta0 = float(angle_str) if angle_str else 0.0  # Keep in radians
            k_opls = float(k_str) if k_str else 0.0

            # Convert energy units only
            # OPLS XML: E = 0.5 * k_opls * (theta - theta0)^2 (kJ/mol, rad)
            # LAMMPS: E = k * (theta - theta0)^2 (kcal/mol, rad)
            # Both use radians, so only energy conversion needed
            # Conversion: k_lammps = 0.5 * k_opls / 4.184
            k = (
                0.5 * k_opls / 4.184
            )  # kJ/mol/rad² to kcal/mol/rad² (accounting for 0.5 factor difference)

            # Define angle type (theta0 in radians)
            anglestyle.def_type(at1, at2, at3, k, theta0)
            count += 1

        logger.info(f"Parsed {count} angle types (OPLS-AA with unit conversion)")

    def _parse_dihedrals(self, element: ET.Element) -> None:
        """
        Parse RBTorsionForce section with OPLS-AA unit conversions.

        Args:
            element: RBTorsionForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        dihedralstyle = self._ff.def_style(DihedralOPLSStyle())
        count = 0

        for dihedral_elem in element:
            if dihedral_elem.tag != "Proper":
                continue

            # Get atom type identifiers
            class1 = dihedral_elem.get("class1", "*")
            class2 = dihedral_elem.get("class2", "*")
            class3 = dihedral_elem.get("class3", "*")
            class4 = dihedral_elem.get("class4", "*")
            type1 = dihedral_elem.get("type1", "*")
            type2 = dihedral_elem.get("type2", "*")
            type3 = dihedral_elem.get("type3", "*")
            type4 = dihedral_elem.get("type4", "*")

            # Check if types are present
            if type1 is None or type2 is None or type3 is None or type4 is None:
                logger.warning("Skipping dihedral without type information")
                continue

            # Get or create atom types
            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)
            at4 = self._get_or_create_atomtype(type4, class4)

            # Parse RB coefficients (c0-c5) from XML
            # OPLS XML stores RB format: E = C0 + C1*cos(phi) + C2*cos²(phi) + C3*cos³(phi) + C4*cos⁴(phi) + C5*cos⁵(phi)
            c0 = float(dihedral_elem.get("c0", "0.0"))  # kJ/mol
            c1 = float(dihedral_elem.get("c1", "0.0"))  # kJ/mol
            c2 = float(dihedral_elem.get("c2", "0.0"))  # kJ/mol
            c3 = float(dihedral_elem.get("c3", "0.0"))  # kJ/mol
            c4 = float(dihedral_elem.get("c4", "0.0"))  # kJ/mol
            c5 = float(dihedral_elem.get("c5", "0.0"))  # kJ/mol

            # Convert RB format to OPLS format (F1-F4) for LAMMPS
            # LAMMPS OPLS dihedral: E = 0.5*(F1*(1+cos(phi)) + F2*(1-cos(2*phi)) + F3*(1+cos(3*phi)) + F4*(1-cos(4*phi)))
            # Conversion formula: F1 = -2*C1 - 3/2*C3, F2 = -C2 - C4, F3 = -1/2*C3, F4 = -1/4*C4
            from molpy.potential.dihedral.opls import rb_to_opls

            k1, k2, k3, k4 = rb_to_opls(c0, c1, c2, c3, c4, c5, units="kJ")
            # k1-k4 are now in kcal/mol (LAMMPS format)

            # Store as c1-c4 for LAMMPS format (c0 and c5 are not used in OPLS format)
            dihedralstyle.def_type(
                at1, at2, at3, at4, c0=0.0, c1=k1, c2=k2, c3=k3, c4=k4, c5=0.0
            )
            count += 1

        logger.info(f"Parsed {count} dihedral types (OPLS-AA with unit conversion)")

    def _parse_nonbonded(self, element: ET.Element) -> None:
        """
        Parse NonbondedForce section with OPLS-AA specific unit conversions.

        Args:
            element: NonbondedForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        # Get scaling factors
        coulomb14scale = element.get("coulomb14scale", "0.5")
        lj14scale = element.get("lj14scale", "0.5")

        # Use lj/cut/coul/long instead of lj/cut/coul/cut for OPLS-AA
        pairstyle = self._ff.def_style(
            PairLJ126CoulLongStyle(
                coulomb14scale=float(coulomb14scale),
                lj14scale=float(lj14scale),
            )
        )
        count = 0

        for atom_elem in element:
            if atom_elem.tag != "Atom":
                continue

            # Get atom type identifier
            type_name = atom_elem.get("type")
            if not type_name:
                logger.warning("Skipping nonbonded atom without type")
                continue

            # Get or create atom type
            atomtype: AtomType | None = None
            if type_name in self._type_to_atomtype:
                atomtype = self._type_to_atomtype[type_name]
            else:
                atomtype = self._get_or_create_atomtype(type_=type_name, class_="*")

            # Parse parameters
            charge_str = atom_elem.get("charge")
            sigma_str = atom_elem.get("sigma")
            epsilon_str = atom_elem.get("epsilon")

            charge = float(charge_str) if charge_str else 0.0
            sigma_opls = float(sigma_str) if sigma_str else 0.0
            epsilon_opls = float(epsilon_str) if epsilon_str else 0.0

            # Convert units for LAMMPS
            # OPLS-AA uses kJ/mol for epsilon, LAMMPS uses kcal/mol
            epsilon_lammps = epsilon_opls / 4.184  # kJ/mol to kcal/mol
            # OPLS-AA uses nm for sigma, LAMMPS uses Angstrom
            sigma_lammps = sigma_opls * 10.0  # nm to Angstrom

            # Define pair parameters (self-interaction)
            pairstyle.def_type(atomtype, atomtype, epsilon_lammps, sigma_lammps, charge)
            count += 1

        logger.info(
            f"Parsed {count} nonbonded parameters (OPLS-AA with unit conversion)"
        )


def read_oplsaa_forcefield(
    filepath: str | Path, forcefield: AtomisticForcefield | None = None
) -> AtomisticForcefield:
    """
    Read OPLS-AA force field with proper unit conversions for LAMMPS.

    This function uses OPLSAAForceFieldReader which:
    - Uses lj/cut/coul/long pair style
    - Converts epsilon: epsilon_lammps = epsilon_opls / 4.184 (kJ/mol to kcal/mol)
    - Converts sigma: sigma_lammps = sigma_opls * 10.0 (nm to Angstrom)

    Args:
        filepath: Path to the OPLS-AA XML file, or "oplsaa.xml" for built-in
        forcefield: Optional existing force field to populate

    Returns:
        Populated AtomisticForcefield object with LAMMPS-compatible units

    Example:
        >>> ff = read_oplsaa_forcefield("oplsaa.xml")
    """
    reader = OPLSAAForceFieldReader(filepath)
    return reader.read(forcefield)
