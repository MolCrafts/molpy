"""XML force field parser for atomistic force fields."""

import logging
import math
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from molpy.core.forcefield import (
    AngleHarmonicStyle,
    AtomisticForcefield,
    AtomType,
    BondHarmonicStyle,
    DihedralOPLSStyle,
    DihedralPeriodicStyle,
    ImproperPeriodicStyle,
    PairLJ126CoulCutStyle,
    PairLJ126CoulLongStyle,
)

logger = logging.getLogger(__name__)

# Angles are stored internally in DEGREES (the molrs convention). A reader
# declares the unit of its *input* file via ``angle_unit`` and converts to
# degrees at the boundary; a writer converts degrees back to its output unit.
ANGLE_UNITS = ("radian", "degree")


def _check_angle_unit(angle_unit: str) -> str:
    if angle_unit not in ANGLE_UNITS:
        raise ValueError(
            f"angle_unit must be one of {ANGLE_UNITS}, got {angle_unit!r}."
        )
    return angle_unit


def _angle_to_internal(value: float, angle_unit: str) -> float:
    """Convert an input angle in *angle_unit* to the internal unit (degrees)."""
    return math.degrees(value) if angle_unit == "radian" else value


def _angle_from_internal(value_deg: float, angle_unit: str) -> float:
    """Convert an internal-degrees angle to *angle_unit* for serialisation."""
    return math.radians(value_deg) if angle_unit == "radian" else value_deg


class AngleUnitWarning(UserWarning):
    """An angle value looks inconsistent with its declared ``angle_unit``."""


# Generous internal-degree sanity bounds per angle kind. These are deliberately
# wide so genuine values (incl. linear 180° angles) never trip them; their job is
# to catch order-of-magnitude unit mismatches (a degree value read as radians
# becomes ~57x larger, e.g. 104.52 -> 5988), not to validate physical chemistry.
_ANGLE_RANGES = {"equilibrium": (0.0, 360.0), "phase": (-360.0, 360.0)}
_TWO_PI = 2.0 * math.pi


def _normalize_angle(raw: float, angle_unit: str, *, kind: str, label: str) -> float:
    """Convert *raw* (in *angle_unit*) to internal degrees, warning on anomalies.

    *kind* selects the plausible range: ``"equilibrium"`` (bond angle ``theta0``,
    0–180°) or ``"phase"`` (dihedral/improper phase, ±360°). An
    :class:`AngleUnitWarning` is emitted when the value is implausible for that
    kind — the usual cause is an ``angle_unit`` that does not match the file
    (e.g. a degree value read as radians, which turned ``104.52`` into ``5988``).

    Args:
        raw: The value as written in the input file.
        angle_unit: The declared unit of *raw* (``"radian"`` or ``"degree"``).
        kind: ``"equilibrium"`` or ``"phase"``.
        label: Human-readable field name for diagnostics (e.g. ``"theta0"``).

    Returns:
        The value in internal degrees.
    """
    if angle_unit == "radian" and abs(raw) > _TWO_PI + 1e-6:
        warnings.warn(
            f"{label}={raw:g} exceeds 2π but angle_unit='radian'; the value looks "
            f"like degrees — pass angle_unit='degree' if the file is in degrees.",
            AngleUnitWarning,
            stacklevel=3,
        )
    deg = _angle_to_internal(raw, angle_unit)
    lo, hi = _ANGLE_RANGES[kind]
    if not (lo - 1e-6 <= deg <= hi + 1e-6):
        warnings.warn(
            f"{label}={deg:g}° is far outside the {kind} sanity bound "
            f"[{lo:g}, {hi:g}]° (angle_unit='{angle_unit}'); likely an angle_unit "
            f"mismatch.",
            AngleUnitWarning,
            stacklevel=3,
        )
    return deg


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

    def __init__(self, filepath: str | Path, *, angle_unit: str = "radian"):
        """
        Initialize the XML force field reader.

        Args:
            filepath: Path to the XML force field file, or filename for built-in files
                     (e.g., "oplsaa.xml" will load from molpy/data/forcefield/)
            angle_unit: Unit of angle equilibria (``"angle"``/phase) in the input
                file — ``"radian"`` (default, the OpenMM/OPLS XML convention) or
                ``"degree"``. Values are converted to the internal degrees
                representation on read, so the rest of the pipeline is unit-consistent.
        """
        self._angle_unit = _check_angle_unit(angle_unit)
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
        # overlay layer for this read: 0 = base force field, >0 = overlay that
        # extends/overrides the base. Atom types created during the read are
        # tagged with this layer so the SMARTS typifier can give overlay
        # patterns strictly higher priority than base ones (CL&P/CL&Pol over
        # OPLS-AA). See _OplsAtomTypifier._extract_patterns.
        self._layer: int = 0

        self._ff: AtomisticForcefield | None = None

    def read(
        self,
        forcefield: AtomisticForcefield | None = None,
        layer: int = 0,
    ) -> AtomisticForcefield:
        """
        Read and parse the XML force field file.

        Args:
            forcefield: Optional existing force field to populate. If None, creates new one.
            layer: Overlay level for this read. ``0`` (default) is the base
                force field; a positive value marks every atom type parsed here
                as an overlay that overrides the base during SMARTS typing
                (higher layer wins). Used to stack ``oplsaa.xml`` (layer 0) →
                ``clp.xml`` (layer 1) → ``clpol.xml`` (layer 2).

        Returns:
            Populated AtomisticForcefield object
        """
        self._layer = layer
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
            elif tag == "PeriodicTorsionForce":
                self._parse_periodic_torsion(child)
            elif tag == "PeriodicImproperForce":
                self._parse_periodic_improper(child)
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

        # Tag overlay-layer types so the typifier can rank them above the base.
        # Only set when reading an overlay (layer > 0) to keep base atom types
        # byte-for-byte identical to the pre-layer behavior.
        if self._layer:
            kwargs.setdefault("layer", self._layer)

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
            bondstyle.def_type(at1, at2, k=k, r0=r0)
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

            # Parse parameters (convert input angle unit -> internal degrees)
            theta0 = (
                _normalize_angle(
                    float(angle_str),
                    self._angle_unit,
                    kind="equilibrium",
                    label="theta0",
                )
                if angle_str
                else 0.0
            )
            k = float(k_str) if k_str else 0.0

            # Define angle type (theta0 in internal degrees)
            anglestyle.def_type(at1, at2, at3, k=k, theta0=theta0)
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
            dihedralstyle.def_type(
                at1, at2, at3, at4, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5
            )
            count += 1

        logger.info(f"Parsed {count} dihedral types")

    def _parse_periodic_torsion(self, element: ET.Element) -> None:
        """
        Parse PeriodicTorsionForce section (periodic cosine dihedrals).

        Args:
            element: PeriodicTorsionForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        dihedralstyle = self._ff.def_style(DihedralPeriodicStyle())
        count = 0

        for dihedral_elem in element:
            if dihedral_elem.tag != "Proper":
                continue

            # Get atom type identifiers
            type1 = dihedral_elem.get("type1", "*")
            type2 = dihedral_elem.get("type2", "*")
            type3 = dihedral_elem.get("type3", "*")
            type4 = dihedral_elem.get("type4", "*")
            class1 = dihedral_elem.get("class1", "*")
            class2 = dihedral_elem.get("class2", "*")
            class3 = dihedral_elem.get("class3", "*")
            class4 = dihedral_elem.get("class4", "*")

            if type1 is None or type2 is None or type3 is None or type4 is None:
                logger.warning("Skipping periodic dihedral without type information")
                continue

            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)
            at4 = self._get_or_create_atomtype(type4, class4)

            # Collect all periodic terms (periodicity1/k1/phase1, periodicity2/k2/phase2, ...)
            params = {}
            for j in range(1, 10):  # support up to 9 terms
                pn = dihedral_elem.get(f"periodicity{j}")
                kn = dihedral_elem.get(f"k{j}")
                phn = dihedral_elem.get(f"phase{j}")
                if pn is not None and kn is not None and phn is not None:
                    params[f"periodicity{j}"] = int(pn)
                    params[f"k{j}"] = float(kn)
                    params[f"phase{j}"] = _normalize_angle(
                        float(phn), self._angle_unit, kind="phase", label=f"phase{j}"
                    )
                else:
                    break

            dihedralstyle.def_type(at1, at2, at3, at4, **params)
            count += 1

        logger.info(f"Parsed {count} periodic dihedral types")

    def _parse_periodic_improper(self, element: ET.Element) -> None:
        """
        Parse PeriodicImproperForce section.

        Args:
            element: PeriodicImproperForce XML element
        """
        assert self._ff is not None, "Force field must be initialized before parsing"
        improperstyle = self._ff.def_style(ImproperPeriodicStyle())
        count = 0

        for imp_elem in element:
            if imp_elem.tag != "Improper":
                continue

            type1 = imp_elem.get("type1", "*")
            type2 = imp_elem.get("type2", "*")
            type3 = imp_elem.get("type3", "*")
            type4 = imp_elem.get("type4", "*")
            class1 = imp_elem.get("class1", "*")
            class2 = imp_elem.get("class2", "*")
            class3 = imp_elem.get("class3", "*")
            class4 = imp_elem.get("class4", "*")

            if type1 is None or type2 is None or type3 is None or type4 is None:
                logger.warning("Skipping periodic improper without type information")
                continue

            at1 = self._get_or_create_atomtype(type1, class1)
            at2 = self._get_or_create_atomtype(type2, class2)
            at3 = self._get_or_create_atomtype(type3, class3)
            at4 = self._get_or_create_atomtype(type4, class4)

            # Collect periodic terms
            params = {}
            for j in range(1, 10):
                pn = imp_elem.get(f"periodicity{j}")
                kn = imp_elem.get(f"k{j}")
                phn = imp_elem.get(f"phase{j}")
                if pn is not None and kn is not None and phn is not None:
                    params[f"periodicity{j}"] = int(pn)
                    params[f"k{j}"] = float(kn)
                    params[f"phase{j}"] = _normalize_angle(
                        float(phn), self._angle_unit, kind="phase", label=f"phase{j}"
                    )
                else:
                    break

            improperstyle.def_type(at1, at2, at3, at4, **params)
            count += 1

        logger.info(f"Parsed {count} periodic improper types")

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
                atomtype, atomtype, epsilon=epsilon, sigma=sigma, charge=charge
            )
            count += 1

        logger.info(f"Parsed {count} nonbonded parameters")


def read_xml_forcefield(
    filepath: str | Path,
    forcefield: AtomisticForcefield | None = None,
    layer: int = 0,
) -> AtomisticForcefield:
    """
    Convenience function to read an XML force field file.

    Args:
        filepath: Path to the XML force field file, or filename for built-in files
                 (e.g., "oplsaa.xml" will load from molpy/data/forcefield/)
        forcefield: Optional existing force field to populate
        layer: Overlay level (0 = base, >0 = overlay that overrides the base
            during SMARTS typing). Pass ``layer=1`` when stacking ``clp.xml``
            onto an already-loaded ``oplsaa.xml`` so CL&P types win conflicts.

    Returns:
        Populated AtomisticForcefield object

    Example:
        >>> # Load built-in OPLS-AA force field
        >>> ff = read_xml_forcefield("oplsaa.xml")
        >>>
        >>> # Overlay CL&P on top of OPLS-AA (CL&P overrides where it matches)
        >>> ff = read_xml_forcefield("clp.xml", ff, layer=1)
    """
    # oplsaa.xml and clp.xml share the OPLS-AA functional form and units, so
    # both are read through the specialized OPLS reader (kJ->kcal, nm->A).
    if str(filepath).endswith(("oplsaa.xml", "clp.xml", "clpol.xml")):
        return read_oplsaa_forcefield(filepath, forcefield, layer=layer)

    reader = XMLForceFieldReader(filepath)
    return reader.read(forcefield, layer=layer)


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
            bondstyle.def_type(at1, at2, k=k, r0=r0)
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
            theta0 = (
                _normalize_angle(
                    float(angle_str),
                    self._angle_unit,
                    kind="equilibrium",
                    label="theta0",
                )
                if angle_str
                else 0.0
            )  # input angle unit -> internal degrees
            k_opls = float(k_str) if k_str else 0.0

            # Convert energy units only (k is per rad², independent of theta0's unit)
            # OPLS XML: E = 0.5 * k_opls * (theta - theta0)^2 (kJ/mol, rad)
            # LAMMPS: E = k * (theta - theta0)^2 (kcal/mol, rad)
            # Conversion: k_lammps = 0.5 * k_opls / 4.184
            k = (
                0.5 * k_opls / 4.184
            )  # kJ/mol/rad² to kcal/mol/rad² (accounting for 0.5 factor difference)

            # Define angle type (theta0 in internal degrees)
            anglestyle.def_type(at1, at2, at3, k=k, theta0=theta0)
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
            from molpy.io.forcefield._rb_opls import rb_to_opls

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
        pairstyle = self._ff.def_pairstyle(
            "lj/cut/coul/long",
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
            pairstyle.def_type(
                atomtype,
                atomtype,
                epsilon=epsilon_lammps,
                sigma=sigma_lammps,
                charge=charge,
            )
            count += 1

        logger.info(
            f"Parsed {count} nonbonded parameters (OPLS-AA with unit conversion)"
        )


def read_oplsaa_forcefield(
    filepath: str | Path,
    forcefield: AtomisticForcefield | None = None,
    layer: int = 0,
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
    return reader.read(forcefield, layer=layer)


# ============================================================================
# Writer
# ============================================================================


class XMLForceFieldWriter:
    """Write a ForceField to OpenMM-style XML.

    The output is roundtrip-compatible with :class:`XMLForceFieldReader`.

    Args:
        filepath: Destination path.
        precision: Number of decimal digits for floating-point values.
    """

    def __init__(
        self, filepath: str | Path, precision: int = 6, *, angle_unit: str = "radian"
    ) -> None:
        self._file = Path(filepath)
        self._prec = precision
        # Output unit for angle equilibria / phases; internal storage is degrees.
        self._angle_unit = _check_angle_unit(angle_unit)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def write(self, forcefield: AtomisticForcefield) -> None:
        """Serialize *forcefield* to XML."""
        from molpy.core.forcefield import (
            AngleStyle,
            AtomStyle,
            BondStyle,
            DihedralStyle,
            ForceField,
            ImproperStyle,
            PairStyle,
        )

        root = ET.Element("ForceField")
        root.set("name", forcefield.name or "MolPy")

        # AtomTypes
        atom_types = forcefield.get_types(AtomType)
        if atom_types:
            self._write_atomtypes(root, atom_types)

        # Bonds — molrs returns styles as their base category class, so dispatch
        # on the style's kernel name, not its (unavailable) specialized class.
        for style in forcefield.get_styles(BondStyle):
            if style.name == "harmonic":
                self._write_harmonic_bonds(root, style)

        # Angles
        for style in forcefield.get_styles(AngleStyle):
            if style.name == "harmonic":
                self._write_harmonic_angles(root, style)

        # Dihedrals
        for style in forcefield.get_styles(DihedralStyle):
            if style.name == "opls":
                self._write_rb_torsions(root, style)
            elif style.name == "periodic":
                self._write_periodic_torsions(root, style)

        # Impropers
        for style in forcefield.get_styles(ImproperStyle):
            if style.name == "periodic":
                self._write_periodic_impropers(root, style)

        # Nonbonded (pairs)
        for style in forcefield.get_styles(PairStyle):
            if style.name in ("lj/cut/coul/cut", "lj/cut/coul/long"):
                self._write_nonbonded(root, style)

        ET.indent(root)
        tree = ET.ElementTree(root)
        tree.write(str(self._file), encoding="unicode", xml_declaration=True)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _fmt(self, v: float | int) -> str:
        if isinstance(v, float):
            return f"{v:.{self._prec}f}"
        return str(v)

    @staticmethod
    def _atom_ident(at: AtomType, idx: int) -> dict[str, str]:
        """Return ``typeN`` / ``classN`` XML attributes for *at*."""
        attrs: dict[str, str] = {}
        type_ = at.params.kwargs.get("type_", "*")
        class_ = at.params.kwargs.get("class_", "*")
        if type_ != "*":
            attrs[f"type{idx}"] = type_
        if class_ != "*":
            attrs[f"class{idx}"] = class_
        return attrs

    # ------------------------------------------------------------------
    # section writers
    # ------------------------------------------------------------------

    def _write_atomtypes(self, root: ET.Element, atom_types: list[AtomType]) -> None:
        section = ET.SubElement(root, "AtomTypes")
        for at in sorted(atom_types, key=lambda t: t.name):
            kw = at.params.kwargs
            attrs: dict[str, str] = {}
            type_ = kw.get("type_", "*")
            class_ = kw.get("class_", "*")
            if type_ != "*":
                attrs["name"] = type_
            if class_ != "*":
                attrs["class"] = class_
            for xml_key, kw_key in [
                ("element", "element"),
                ("mass", "mass"),
                ("def", "def_"),
                ("desc", "desc"),
                ("doi", "doi"),
                ("overrides", "overrides"),
            ]:
                val = kw.get(kw_key)
                if val is not None:
                    attrs[xml_key] = (
                        self._fmt(val) if isinstance(val, float) else str(val)
                    )
            elem = ET.SubElement(section, "Type")
            for k, v in attrs.items():
                elem.set(k, v)

    def _write_harmonic_bonds(self, root: ET.Element, style: BondHarmonicStyle) -> None:
        from molpy.core.forcefield import BondType

        types = list(style.get_types(BondType))
        if not types:
            return
        section = ET.SubElement(root, "HarmonicBondForce")
        for bt in types:
            attrs = {**self._atom_ident(bt.itom, 1), **self._atom_ident(bt.jtom, 2)}
            kw = bt.params.kwargs
            r0 = kw.get("r0", 0.0)
            k = kw.get("k", 0.0)
            attrs["length"] = self._fmt(r0)
            attrs["k"] = self._fmt(k)
            elem = ET.SubElement(section, "Bond")
            for a, v in attrs.items():
                elem.set(a, v)

    def _write_harmonic_angles(
        self, root: ET.Element, style: AngleHarmonicStyle
    ) -> None:
        from molpy.core.forcefield import AngleType

        types = list(style.get_types(AngleType))
        if not types:
            return
        section = ET.SubElement(root, "HarmonicAngleForce")
        for at in types:
            attrs = {
                **self._atom_ident(at.itom, 1),
                **self._atom_ident(at.jtom, 2),
                **self._atom_ident(at.ktom, 3),
            }
            kw = at.params.kwargs
            attrs["angle"] = self._fmt(
                _angle_from_internal(kw.get("theta0", 0.0), self._angle_unit)
            )
            attrs["k"] = self._fmt(kw.get("k", 0.0))
            elem = ET.SubElement(section, "Angle")
            for a, v in attrs.items():
                elem.set(a, v)

    def _write_rb_torsions(self, root: ET.Element, style: DihedralOPLSStyle) -> None:
        from molpy.core.forcefield import DihedralType

        types = list(style.get_types(DihedralType))
        if not types:
            return
        section = ET.SubElement(root, "RBTorsionForce")
        for dt in types:
            attrs = {
                **self._atom_ident(dt.itom, 1),
                **self._atom_ident(dt.jtom, 2),
                **self._atom_ident(dt.ktom, 3),
                **self._atom_ident(dt.ltom, 4),
            }
            kw = dt.params.kwargs
            for ci in range(6):
                attrs[f"c{ci}"] = self._fmt(kw.get(f"c{ci}", 0.0))
            elem = ET.SubElement(section, "Proper")
            for a, v in attrs.items():
                elem.set(a, v)

    def _write_periodic_torsions(
        self, root: ET.Element, style: DihedralPeriodicStyle
    ) -> None:
        from molpy.core.forcefield import DihedralType

        types = list(style.get_types(DihedralType))
        if not types:
            return
        section = ET.SubElement(root, "PeriodicTorsionForce")
        for dt in types:
            attrs = {
                **self._atom_ident(dt.itom, 1),
                **self._atom_ident(dt.jtom, 2),
                **self._atom_ident(dt.ktom, 3),
                **self._atom_ident(dt.ltom, 4),
            }
            kw = dt.params.kwargs
            for j in range(1, 10):
                pk = f"periodicity{j}"
                kk = f"k{j}"
                phk = f"phase{j}"
                if pk in kw and kk in kw and phk in kw:
                    attrs[pk] = str(int(kw[pk]))
                    attrs[kk] = self._fmt(kw[kk])
                    attrs[phk] = self._fmt(
                        _angle_from_internal(kw[phk], self._angle_unit)
                    )
                else:
                    break
            elem = ET.SubElement(section, "Proper")
            for a, v in attrs.items():
                elem.set(a, v)

    def _write_periodic_impropers(
        self, root: ET.Element, style: ImproperPeriodicStyle
    ) -> None:
        from molpy.core.forcefield import ImproperType

        types = list(style.get_types(ImproperType))
        if not types:
            return
        section = ET.SubElement(root, "PeriodicImproperForce")
        for it in types:
            attrs = {
                **self._atom_ident(it.itom, 1),
                **self._atom_ident(it.jtom, 2),
                **self._atom_ident(it.ktom, 3),
                **self._atom_ident(it.ltom, 4),
            }
            kw = it.params.kwargs
            for j in range(1, 10):
                pk = f"periodicity{j}"
                kk = f"k{j}"
                phk = f"phase{j}"
                if pk in kw and kk in kw and phk in kw:
                    attrs[pk] = str(int(kw[pk]))
                    attrs[kk] = self._fmt(kw[kk])
                    attrs[phk] = self._fmt(
                        _angle_from_internal(kw[phk], self._angle_unit)
                    )
                else:
                    break
            elem = ET.SubElement(section, "Improper")
            for a, v in attrs.items():
                elem.set(a, v)

    def _write_nonbonded(
        self, root: ET.Element, style: PairLJ126CoulCutStyle | PairLJ126CoulLongStyle
    ) -> None:
        from molpy.core.forcefield import PairType

        types = list(style.get_types(PairType))
        if not types:
            return
        section = ET.SubElement(root, "NonbondedForce")
        skw = style._ff.style_params(style.category, style.name) or {}
        section.set("coulomb14scale", self._fmt(skw.get("coulomb14scale", 0.5)))
        section.set("lj14scale", self._fmt(skw.get("lj14scale", 0.5)))

        for pt in types:
            kw = pt.params.kwargs
            type_name = pt.itom.params.kwargs.get("type_", pt.itom.name)
            if type_name == "*":
                type_name = pt.itom.name
            attrs: dict[str, str] = {"type": type_name}
            for xml_key, kw_key in [
                ("charge", "charge"),
                ("sigma", "sigma"),
                ("epsilon", "epsilon"),
            ]:
                val = kw.get(kw_key)
                if val is not None:
                    attrs[xml_key] = self._fmt(val)
            elem = ET.SubElement(section, "Atom")
            for a, v in attrs.items():
                elem.set(a, v)


def write_xml_forcefield(filepath: str | Path, forcefield: AtomisticForcefield) -> None:
    """Convenience function to write a force field to XML.

    Args:
        filepath: Output path.
        forcefield: Force field to serialize.
    """
    XMLForceFieldWriter(filepath).write(forcefield)
