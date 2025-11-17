"""XML force field parser for atomistic force fields."""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from molpy import (
    AtomisticForcefield,
    AtomType,
    DihedralType,
)

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
                assert self._ff is not None, (
                    "Force field must be initialized before creating atom types"
                )
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
            assert self._ff is not None, (
                "Force field must be initialized before creating atom types"
            )
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
            assert self._ff is not None, (
                "Force field must be initialized before creating atom types"
            )
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
        assert self._ff is not None, (
            "Force field must be initialized before creating atom types"
        )
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
        bondstyle = self._ff.def_bondstyle("harmonic")
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
            bondstyle.def_type(at1, at2, r0=r0, k=k)
            count += 1

        logger.info(f"Parsed {count} bond types")

    def _parse_angles(self, element: ET.Element) -> None:
        """
        Parse HarmonicAngleForce section.

        Args:
            element: HarmonicAngleForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        anglestyle = self._ff.def_anglestyle("harmonic")
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
            theta0 = float(angle_str) if angle_str else 0.0
            k = float(k_str) if k_str else 0.0

            # Define angle type
            anglestyle.def_type(at1, at2, at3, theta0=theta0, k=k)
            count += 1

        logger.info(f"Parsed {count} angle types")

    def _parse_dihedrals(self, element: ET.Element) -> None:
        """
        Parse RBTorsionForce section (Ryckaert-Bellemans dihedrals).

        Args:
            element: RBTorsionForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        dihedralstyle = self._ff.def_dihedralstyle("opls")
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
            params = {}
            for i in range(6):
                c_str = dihedral_elem.get(f"c{i}")
                if c_str:
                    params[f"c{i}"] = float(c_str)

            # Define dihedral type - create and add manually
            # Use more specific name: if type is "*", use class name instead
            name1 = type1 if type1 != "*" else class1
            name2 = type2 if type2 != "*" else class2
            name3 = type3 if type3 != "*" else class3
            name4 = type4 if type4 != "*" else class4
            dihedral_name = f"{name1}-{name2}-{name3}-{name4}"

            # If still not unique (same name already exists), append counter
            # Note: we check by trying to find existing type with same name
            base_name = dihedral_name
            counter = 1
            existing_names = {
                dt.name for dt in dihedralstyle.types.bucket(DihedralType)
            }
            while dihedral_name in existing_names:
                dihedral_name = f"{base_name}#{counter}"
                counter += 1

            dihedral_type = DihedralType(dihedral_name, at1, at2, at3, at4, **params)
            dihedralstyle.types.add(dihedral_type)
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

        pairstyle = self._ff.def_pairstyle(
            "lj/cut/coul/cut",
            coulomb14scale=float(coulomb14scale),
            lj14scale=float(lj14scale),
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
            pairstyle.def_type(
                atomtype, atomtype, charge=charge, sigma=sigma, epsilon=epsilon
            )
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
    reader = XMLForceFieldReader(filepath)
    return reader.read(forcefield)
