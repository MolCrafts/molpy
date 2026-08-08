#!/usr/bin/env python3
"""Unit tests for XML force field reader.

This module contains comprehensive tests for:
- XML force field file reading functionality
- Force field parameter extraction and validation
- Atom types, bond types, angle types, dihedral types, and pair types
- Error handling and edge cases

Uses pytest framework with modern Python 3.10+ type hints and Google-style docstrings.
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from molpy import AngleType, ForceField, AtomType, BondType, PairType
from molpy.data import get_forcefield_path
from molpy.io.forcefield.xml import XMLForceFieldReader, read_xml_forcefield


class TestXMLForceFieldReader:
    """Test suite for XML force field reader."""

    def test_read_pf6_xml_forcefield(self, TEST_DATA_DIR: Path) -> None:
        """Test reading PF6 XML force field file and validate all parameters.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        xml_file = TEST_DATA_DIR / "xml" / "pf6.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        # Parse XML file directly for comparison
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Read force field using reader
        reader = XMLForceFieldReader(xml_file)
        ff = reader.read()

        # Validate force field metadata
        assert ff.name == root.get("name", "Unknown")
        assert ff.units == "real"

        # Parse and validate AtomTypes
        atomtypes_elem = root.find("AtomTypes")
        if atomtypes_elem is not None:
            expected_atomtypes = {}
            for type_elem in atomtypes_elem.findall("Type"):
                name = type_elem.get("name")
                class_ = type_elem.get("class")
                element = type_elem.get("element")
                mass = type_elem.get("mass")
                def_ = type_elem.get("def")
                desc = type_elem.get("desc")
                overrides = type_elem.get("overrides")

                expected_atomtypes[name] = {
                    "name": name,
                    "class": class_,
                    "element": element,
                    "mass": float(mass) if mass else None,
                    "def": def_,
                    "desc": desc,
                    "overrides": overrides,
                }

            # Validate parsed atom types.
            # The molrs-backed reader may also create extra wildcard/class-based
            # atom types while parsing bonds/pairs, so the declared types are a
            # subset (>=) rather than an exact count.
            atomtypes = ff.get_types(AtomType)
            assert len(atomtypes) >= len(expected_atomtypes), (
                f"Expected at least {len(expected_atomtypes)} atom types, got {len(atomtypes)}"
            )

            for name, expected in expected_atomtypes.items():
                # Find the declared atom type by name and matching metadata.
                found = None
                for at in atomtypes:
                    if at.name == name and at.params.kwargs.get("type_") != "*":
                        found = at
                        break
                if found is None:
                    for at in atomtypes:
                        if at.name == name:
                            found = at
                            break

                assert found is not None, f"Atom type '{name}' not found"
                assert (
                    found.params.kwargs.get("type_") == expected["name"]
                    or found.name == expected["name"]
                )
                assert (
                    found.params.kwargs.get("class_") == expected["class"]
                    or found.name == expected["class"]
                )
                if expected["element"]:
                    assert found.params.kwargs.get("element") == expected["element"]
                if expected["mass"]:
                    assert (
                        abs(found.params.kwargs.get("mass", 0) - expected["mass"])
                        < 1e-6
                    )
                if expected["def"]:
                    assert found.params.kwargs.get("def_") == expected["def"]
                if expected["desc"]:
                    assert found.params.kwargs.get("desc") == expected["desc"]
                if expected["overrides"]:
                    assert found.params.kwargs.get("overrides") == expected["overrides"]

        # Parse and validate NonbondedForce
        nonbonded_elem = root.find("NonbondedForce")
        if nonbonded_elem is not None:
            expected_pairs = {}
            for atom_elem in nonbonded_elem.findall("Atom"):
                type_name = atom_elem.get("type")
                charge = atom_elem.get("charge")
                sigma = atom_elem.get("sigma")
                epsilon = atom_elem.get("epsilon")

                expected_pairs[type_name] = {
                    "charge": float(charge) if charge else None,
                    "sigma": float(sigma) if sigma else None,
                    "epsilon": float(epsilon) if epsilon else None,
                }

            # Validate parsed pair types
            pairtypes = ff.get_types(PairType)
            assert len(pairtypes) >= len(expected_pairs), (
                f"Expected at least {len(expected_pairs)} pair types, got {len(pairtypes)}"
            )

            for type_name, expected in expected_pairs.items():
                # Find pair type by atom type name (self-pair name == type name).
                found = None
                for pt in pairtypes:
                    if (
                        pt.name == type_name
                        or pt.itom.name == type_name
                        or pt.itom.params.kwargs.get("type_") == type_name
                    ):
                        found = pt
                        break

                assert found is not None, f"Pair type for '{type_name}' not found"
                # Charge lives on the AtomType (molrs OPLS path); pair rows only
                # carry LJ. Accept either location for charge.
                if expected["charge"] is not None:
                    pair_q = found.params.kwargs.get("charge")
                    atom_q = None
                    for at in ff.get_types(AtomType):
                        if (
                            at.name == type_name
                            and at.params.kwargs.get("type_") != "*"
                        ):
                            atom_q = at.params.kwargs.get("charge")
                            break
                    got_q = pair_q if pair_q is not None else atom_q
                    assert got_q is not None, (
                        f"charge for {type_name} not on pair or atom"
                    )
                    assert abs(got_q - expected["charge"]) < 1e-6
                if expected["sigma"] is not None:
                    s = found.params.kwargs.get("sigma", 0)
                    # OpenMM σ is nm; molrs OPLS path → Å (×10).
                    assert (
                        abs(s - expected["sigma"]) < 1e-6
                        or abs(s - expected["sigma"] * 10.0) < 1e-5
                    )
                if expected["epsilon"] is not None:
                    e = found.params.kwargs.get("epsilon", 0)
                    assert (
                        abs(e - expected["epsilon"]) < 1e-6
                        or abs(e - expected["epsilon"] / 4.184) < 1e-5
                    )

        # Parse and validate HarmonicBondForce
        bonds_elem = root.find("HarmonicBondForce")
        if bonds_elem is not None:
            expected_bonds = []
            for bond_elem in bonds_elem.findall("Bond"):
                class1 = bond_elem.get("class1", "*")
                class2 = bond_elem.get("class2", "*")
                type1 = bond_elem.get("type1", "*")
                type2 = bond_elem.get("type2", "*")
                length = bond_elem.get("length")
                k = bond_elem.get("k")

                expected_bonds.append(
                    {
                        "class1": class1,
                        "class2": class2,
                        "type1": type1,
                        "type2": type2,
                        "length": float(length) if length else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed bond types
            bondtypes = ff.get_types(BondType)
            assert len(bondtypes) == len(expected_bonds), (
                f"Expected {len(expected_bonds)} bond types, got {len(bondtypes)}"
            )

            for expected in expected_bonds:
                # Find bond type by atom types.
                # molrs drops the BondType endpoint params, but preserves the
                # endpoint AtomType *name* (which equals the class for
                # class-based bonds), so match on name against type/class too.
                found = None
                for bt in bondtypes:
                    at1_match = (
                        bt.itom.name == expected["type1"]
                        or bt.itom.name == expected["class1"]
                        or bt.itom.params.kwargs.get("type_") == expected["type1"]
                        or bt.itom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_match = (
                        bt.jtom.name == expected["type2"]
                        or bt.jtom.name == expected["class2"]
                        or bt.jtom.params.kwargs.get("type_") == expected["type2"]
                        or bt.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    if at1_match and at2_match:
                        found = bt
                        break

                assert found is not None, (
                    f"Bond type '{expected['type1']}-{expected['type2']}' not found"
                )
                if expected["length"] is not None:
                    r0 = found.params.kwargs.get("r0", 0)
                    # OpenMM length is nm; molrs OPLS path → Å (×10).
                    assert (
                        abs(r0 - expected["length"]) < 1e-6
                        or abs(r0 - expected["length"] * 10.0) < 1e-5
                    )
                if expected["k"] is not None:
                    k_got = found.params.kwargs.get(
                        "k0", found.params.kwargs.get("k", 0)
                    )
                    k_kcal = expected["k"] / (4.184 * 100.0)
                    assert (
                        abs(k_got - expected["k"]) < 1e-3 or abs(k_got - k_kcal) < 1e-2
                    )

        # Parse and validate HarmonicAngleForce
        angles_elem = root.find("HarmonicAngleForce")
        if angles_elem is not None:
            expected_angles = []
            for angle_elem in angles_elem.findall("Angle"):
                class1 = angle_elem.get("class1", "*")
                class2 = angle_elem.get("class2", "*")
                class3 = angle_elem.get("class3", "*")
                type1 = angle_elem.get("type1", "*")
                type2 = angle_elem.get("type2", "*")
                type3 = angle_elem.get("type3", "*")
                angle = angle_elem.get("angle")
                k = angle_elem.get("k")

                expected_angles.append(
                    {
                        "class1": class1,
                        "class2": class2,
                        "class3": class3,
                        "type1": type1,
                        "type2": type2,
                        "type3": type3,
                        # XML stores radians, and radians are the internal unit.
                        "angle": float(angle) if angle else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed angle types
            angletypes = ff.get_types(AngleType)
            assert len(angletypes) == len(expected_angles), (
                f"Expected {len(expected_angles)} angle types, got {len(angletypes)}"
            )

            # Track which angles we've matched
            matched_indices = set()
            for expected in expected_angles:
                # Find angle type by atom types (try both forward and reverse)
                found = None
                found_idx = None
                for idx, at in enumerate(angletypes):
                    if idx in matched_indices:
                        continue
                    # Forward match (molrs preserves endpoint name == class for
                    # class-based angles, so match on name against type/class).
                    at1_match = (
                        at.itom.name == expected["type1"]
                        or at.itom.name == expected["class1"]
                        or at.itom.params.kwargs.get("type_") == expected["type1"]
                        or at.itom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_match = (
                        at.jtom.name == expected["type2"]
                        or at.jtom.name == expected["class2"]
                        or at.jtom.params.kwargs.get("type_") == expected["type2"]
                        or at.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    at3_match = (
                        at.ktom.name == expected["type3"]
                        or at.ktom.name == expected["class3"]
                        or at.ktom.params.kwargs.get("type_") == expected["type3"]
                        or at.ktom.params.kwargs.get("class_") == expected["class3"]
                    )
                    # Reverse match
                    at1_rev_match = (
                        at.ktom.name == expected["type1"]
                        or at.ktom.name == expected["class1"]
                        or at.ktom.params.kwargs.get("type_") == expected["type1"]
                        or at.ktom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_rev_match = (
                        at.jtom.name == expected["type2"]
                        or at.jtom.name == expected["class2"]
                        or at.jtom.params.kwargs.get("type_") == expected["type2"]
                        or at.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    at3_rev_match = (
                        at.itom.name == expected["type3"]
                        or at.itom.name == expected["class3"]
                        or at.itom.params.kwargs.get("type_") == expected["type3"]
                        or at.itom.params.kwargs.get("class_") == expected["class3"]
                    )

                    if (at1_match and at2_match and at3_match) or (
                        at1_rev_match and at2_rev_match and at3_rev_match
                    ):
                        # Check if parameters match
                        if expected["angle"] is not None:
                            theta0 = (
                                found.params.kwargs.get("theta0", 0)
                                if found
                                else at.params.kwargs.get("theta0", 0)
                            )
                            if abs(theta0 - expected["angle"]) < 1e-6:
                                found = at
                                found_idx = idx
                                break
                        elif found is None:
                            # If no angle specified, take first match
                            found = at
                            found_idx = idx
                            break

                assert found is not None, (
                    f"Angle type '{expected['type1']}-{expected['type2']}-{expected['type3']}' not found"
                )
                if found_idx is not None:
                    matched_indices.add(found_idx)
                if expected["angle"] is not None:
                    assert (
                        abs(found.params.kwargs.get("theta0", 0) - expected["angle"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    k_got = found.params.kwargs.get(
                        "k0", found.params.kwargs.get("k", 0)
                    )
                    k_kcal = expected["k"] / 4.184
                    assert (
                        abs(k_got - expected["k"]) < 1e-3 or abs(k_got - k_kcal) < 1e-2
                    )

    def test_read_tip3p_xml_forcefield(self, TEST_DATA_DIR: Path) -> None:
        """Test reading TIP3P XML force field file and validate all parameters.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        xml_file = TEST_DATA_DIR / "xml" / "tip3p.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        # Parse XML file directly for comparison
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Read force field using reader
        reader = XMLForceFieldReader(xml_file)
        ff = reader.read()

        # Validate force field metadata
        assert ff.name == root.get("name", "Unknown")
        assert ff.units == "real"

        # Parse and validate AtomTypes
        atomtypes_elem = root.find("AtomTypes")
        if atomtypes_elem is not None:
            expected_atomtypes = {}
            for type_elem in atomtypes_elem.findall("Type"):
                name = type_elem.get("name")
                class_ = type_elem.get("class")
                element = type_elem.get("element")
                mass = type_elem.get("mass")
                def_ = type_elem.get("def")

                expected_atomtypes[name] = {
                    "name": name,
                    "class": class_,
                    "element": element,
                    "mass": float(mass) if mass else None,
                    "def": def_,
                }

            # Validate parsed atom types
            atomtypes = ff.get_types(AtomType)
            assert len(atomtypes) == len(expected_atomtypes), (
                f"Expected {len(expected_atomtypes)} atom types, got {len(atomtypes)}"
            )

            for name, expected in expected_atomtypes.items():
                # Find atom type by name
                found = None
                for at in atomtypes:
                    if at.name == name:
                        found = at
                        break

                assert found is not None, f"Atom type '{name}' not found"
                assert (
                    found.params.kwargs.get("type_") == expected["name"]
                    or found.name == expected["name"]
                )
                assert (
                    found.params.kwargs.get("class_") == expected["class"]
                    or found.name == expected["class"]
                )
                if expected["element"]:
                    assert found.params.kwargs.get("element") == expected["element"]
                if expected["mass"]:
                    assert (
                        abs(found.params.kwargs.get("mass", 0) - expected["mass"])
                        < 1e-6
                    )
                if expected["def"]:
                    assert found.params.kwargs.get("def_") == expected["def"]

        # Parse and validate HarmonicBondForce
        bonds_elem = root.find("HarmonicBondForce")
        if bonds_elem is not None:
            expected_bonds = []
            for bond_elem in bonds_elem.findall("Bond"):
                type1 = bond_elem.get("type1", "*")
                type2 = bond_elem.get("type2", "*")
                length = bond_elem.get("length")
                k = bond_elem.get("k")

                expected_bonds.append(
                    {
                        "type1": type1,
                        "type2": type2,
                        "length": float(length) if length else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed bond types
            bondtypes = ff.get_types(BondType)
            assert len(bondtypes) == len(expected_bonds), (
                f"Expected {len(expected_bonds)} bond types, got {len(bondtypes)}"
            )

            for expected in expected_bonds:
                # Find bond type by atom types
                found = None
                for bt in bondtypes:
                    at1_match = (
                        bt.itom.name == expected["type1"]
                        or bt.itom.params.kwargs.get("type_") == expected["type1"]
                    )
                    at2_match = (
                        bt.jtom.name == expected["type2"]
                        or bt.jtom.params.kwargs.get("type_") == expected["type2"]
                    )
                    if at1_match and at2_match:
                        found = bt
                        break

                assert found is not None, (
                    f"Bond type '{expected['type1']}-{expected['type2']}' not found"
                )
                if expected["length"] is not None:
                    # OpenMM packs store length in nm; molrs OPLS path → Å (×10).
                    r0 = found.params.kwargs.get("r0", 0)
                    assert (
                        abs(r0 - expected["length"]) < 1e-6
                        or abs(r0 - expected["length"] * 10.0) < 1e-5
                    )
                if expected["k"] is not None:
                    # OpenMM k is kJ/mol/nm²; molrs stores kcal/mol/Å² as k0.
                    k_got = found.params.kwargs.get(
                        "k0", found.params.kwargs.get("k", 0)
                    )
                    k_kcal = expected["k"] / (4.184 * 100.0)
                    assert (
                        abs(k_got - expected["k"]) < 1e-3 or abs(k_got - k_kcal) < 1e-2
                    )

        # Parse and validate HarmonicAngleForce
        angles_elem = root.find("HarmonicAngleForce")
        if angles_elem is not None:
            expected_angles = []
            for angle_elem in angles_elem.findall("Angle"):
                type1 = angle_elem.get("type1", "*")
                type2 = angle_elem.get("type2", "*")
                type3 = angle_elem.get("type3", "*")
                angle = angle_elem.get("angle")
                k = angle_elem.get("k")

                expected_angles.append(
                    {
                        "type1": type1,
                        "type2": type2,
                        "type3": type3,
                        # XML stores radians, and radians are the internal unit.
                        "angle": float(angle) if angle else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed angle types
            angletypes = ff.get_types(AngleType)
            assert len(angletypes) == len(expected_angles), (
                f"Expected {len(expected_angles)} angle types, got {len(angletypes)}"
            )

            for expected in expected_angles:
                # Find angle type by atom types
                found = None
                for at in angletypes:
                    at1_match = (
                        at.itom.name == expected["type1"]
                        or at.itom.params.kwargs.get("type_") == expected["type1"]
                    )
                    at2_match = (
                        at.jtom.name == expected["type2"]
                        or at.jtom.params.kwargs.get("type_") == expected["type2"]
                    )
                    at3_match = (
                        at.ktom.name == expected["type3"]
                        or at.ktom.params.kwargs.get("type_") == expected["type3"]
                    )
                    if at1_match and at2_match and at3_match:
                        found = at
                        break

                assert found is not None, (
                    f"Angle type '{expected['type1']}-{expected['type2']}-{expected['type3']}' not found"
                )
                if expected["angle"] is not None:
                    assert (
                        abs(found.params.kwargs.get("theta0", 0) - expected["angle"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    # OpenMM k is kJ/mol/rad²; molrs stores kcal/mol/rad² as k0.
                    k_got = found.params.kwargs.get(
                        "k0", found.params.kwargs.get("k", 0)
                    )
                    k_kcal = expected["k"] / 4.184
                    assert (
                        abs(k_got - expected["k"]) < 1e-3 or abs(k_got - k_kcal) < 1e-2
                    )

        # Parse and validate NonbondedForce
        nonbonded_elem = root.find("NonbondedForce")
        if nonbonded_elem is not None:
            expected_pairs = {}
            for atom_elem in nonbonded_elem.findall("Atom"):
                type_name = atom_elem.get("type")
                sigma = atom_elem.get("sigma")
                epsilon = atom_elem.get("epsilon")

                if type_name:
                    expected_pairs[type_name] = {
                        "sigma": float(sigma) if sigma else None,
                        "epsilon": float(epsilon) if epsilon else None,
                    }

            # Validate parsed pair types
            pairtypes = ff.get_types(PairType)
            assert len(pairtypes) >= len(expected_pairs), (
                f"Expected at least {len(expected_pairs)} pair types, got {len(pairtypes)}"
            )

            for type_name, expected in expected_pairs.items():
                # Find pair type by atom type name (self-pair name == type name).
                found = None
                for pt in pairtypes:
                    if (
                        pt.name == type_name
                        or pt.itom.name == type_name
                        or pt.itom.params.kwargs.get("type_") == type_name
                    ):
                        found = pt
                        break

                assert found is not None, f"Pair type for '{type_name}' not found"
                if expected["sigma"] is not None:
                    # OpenMM σ is nm; molrs OPLS path → Å (×10).
                    s = found.params.kwargs.get("sigma", 0)
                    assert (
                        abs(s - expected["sigma"]) < 1e-6
                        or abs(s - expected["sigma"] * 10.0) < 1e-5
                    )
                if expected["epsilon"] is not None:
                    # OpenMM ε is kJ/mol; molrs stores kcal/mol (÷4.184).
                    e = found.params.kwargs.get("epsilon", 0)
                    assert (
                        abs(e - expected["epsilon"]) < 1e-6
                        or abs(e - expected["epsilon"] / 4.184) < 1e-5
                    )

    def test_read_oplsaa_xml_forcefield_metadata(self, TEST_DATA_DIR: Path) -> None:
        """Test reading OPLS-AA XML force field file metadata.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        xml_file = TEST_DATA_DIR / "xml" / "oplsaa.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        # Parse XML file directly for comparison
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Read force field using reader
        reader = XMLForceFieldReader(xml_file)
        ff = reader.read()

        # Validate force field metadata
        expected_name = root.get("name", "Unknown")
        root.get("version", "0.0.0")
        root.get("combining_rule", "geometric")

        assert ff.name == expected_name
        assert ff.units == "real"

        # Parse and validate AtomTypes count
        # Note: The parser may create additional atom types (e.g., wildcard types, class-based types)
        # So we check that we have at least the expected number
        atomtypes_elem = root.find("AtomTypes")
        if atomtypes_elem is not None:
            expected_count = len(list(atomtypes_elem.findall("Type")))
            atomtypes = ff.get_types(AtomType)
            # The parser may create additional types, so we check >=
            assert len(atomtypes) >= expected_count, (
                f"Expected at least {expected_count} atom types, got {len(atomtypes)}"
            )

            # But we should verify all expected types are present
            expected_names = {
                type_elem.get("name") for type_elem in atomtypes_elem.findall("Type")
            }
            actual_names = {at.name for at in atomtypes}
            missing = expected_names - actual_names
            assert len(missing) == 0, f"Missing atom types: {missing}"

    def test_file_not_found_error(self) -> None:
        """Test that FileNotFoundError is raised for nonexistent files."""
        with pytest.raises(FileNotFoundError):
            read_xml_forcefield("nonexistent_forcefield.xml")

    def test_read_xml_forcefield_convenience_function(
        self, TEST_DATA_DIR: Path
    ) -> None:
        """Test the convenience function read_xml_forcefield.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        xml_file = TEST_DATA_DIR / "xml" / "pf6.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        ff = read_xml_forcefield(xml_file)
        assert isinstance(ff, ForceField)
        assert len(ff.get_types(AtomType)) > 0

    def test_read_xml_forcefield_with_existing_forcefield(
        self, TEST_DATA_DIR: Path
    ) -> None:
        """Test reading XML into an existing force field.

        Args:
            TEST_DATA_DIR: Path to test data directory fixture.
        """
        xml_file = TEST_DATA_DIR / "xml" / "pf6.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        # Create existing force field
        existing_ff = ForceField(name="test", units="real")

        # Read XML into existing force field
        ff = read_xml_forcefield(xml_file, forcefield=existing_ff)

        # Should be the same object
        assert ff is existing_ff
        assert len(ff.get_types(AtomType)) > 0

    def test_parse_bond_with_class_only_creates_wildcard_atomtypes(
        self, tmp_path: Path
    ) -> None:
        """Class-only bond endpoints create wildcard AtomTypes (type="*", class=…).

        A three-element fixture is enough — loading full ``oplsaa.xml`` and
        walking every AtomType.params was multi-second noise.
        """
        xml = tmp_path / "class_bond.xml"
        xml.write_text(
            """<?xml version="1.0"?>
<ForceField>
  <AtomTypes>
    <Type name="opls_c" class="C" element="C" mass="12.01"/>
    <Type name="opls_o" class="O_3" element="O" mass="16.00"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="C" class2="O_3" length="0.14" k="1000.0"/>
  </HarmonicBondForce>
</ForceField>
"""
        )
        ff = read_xml_forcefield(xml)

        wildcards = {
            (
                at.params.kwargs.get("class_", ""),
                at.params.kwargs.get("type_", ""),
            ): at
            for at in ff.get_types(AtomType)
        }
        assert ("O_3", "*") in wildcards
        assert ("C", "*") in wildcards
        assert wildcards[("O_3", "*")].name == "O_3"
        assert wildcards[("C", "*")].name == "C"

        # molrs rebuilds endpoint AtomTypes; class-only wildcards use name==class.
        bond_found = False
        for bt in ff.get_types(BondType):
            if {bt.itom.name, bt.jtom.name} == {"C", "O_3"}:
                bond_found = True
                break
        assert bond_found, "C - O_3 bond type should exist and use wildcard AtomTypes"

    def test_class_based_bond_typing_works(self, tmp_path: Path) -> None:
        """Bonds type via class match once wildcard AtomTypes exist."""
        from molpy import Atomistic
        from molpy.typifier import ForceFieldParams

        xml = tmp_path / "class_bond.xml"
        xml.write_text(
            """<?xml version="1.0"?>
<ForceField>
  <AtomTypes>
    <Type name="opls_267" class="C" element="C" mass="12.01"/>
    <Type name="opls_269" class="O_3" element="O" mass="16.00"/>
  </AtomTypes>
  <HarmonicBondForce>
    <Bond class1="C" class2="O_3" length="0.14" k="250000.0"/>
  </HarmonicBondForce>
  <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
    <Atom type="opls_267" charge="0.0" sigma="0.35" epsilon="0.3"/>
    <Atom type="opls_269" charge="-0.5" sigma="0.3" epsilon="0.7"/>
  </NonbondedForce>
</ForceField>
"""
        )
        ff = read_xml_forcefield(xml)

        asm = Atomistic()
        atom1 = asm.def_atom(symbol="O", type="opls_269")  # class="O_3"
        atom2 = asm.def_atom(symbol="C", type="opls_267")  # class="C"
        bond = asm.def_bond(atom1, atom2)

        typed = ForceFieldParams(ff, strict=False).assign(asm)
        typed_bond = next(iter(typed.bonds))

        assert typed_bond.get("type") is not None, "Bond should have a type assigned"
        assert "k" in typed_bond.data or "r0" in typed_bond.data, (
            "Bond should have parameters"
        )
        assert bond.data == {}, "assign must not mutate its input"


class TestAngleUnitOption:
    """Input angle unit is configurable; internal storage is always radians."""

    def _theta0_rad(self, ff):
        from molpy import AngleStyle

        for style in ff.get_styles(AngleStyle):
            for typ in style.types:
                v = typ.params.kwargs.get("theta0")
                if v:
                    return v
        return None

    def test_radian_input_is_kept_as_the_internal_unit(self):
        """The default (radian) XML input needs no conversion — radians are internal.

        This asserted degrees, which is what let the reader ship a 104.52 that
        molrs's LAMMPS writer then multiplied by 180/π into 5988.55.
        """
        ff = XMLForceFieldReader(
            get_forcefield_path("oplsaa.xml"), angle_unit="radian"
        ).read()
        theta0 = self._theta0_rad(ff)
        assert theta0 is not None
        assert 1.4 < theta0 < math.pi  # radians — the molrs internal unit

    def test_degree_input_is_converted_and_radian_input_is_not(self):
        """A degree file converts at the boundary; a radian file passes through."""
        from molpy.io.forcefield.xml import _angle_to_internal

        assert abs(_angle_to_internal(109.5, "degree") - math.radians(109.5)) < 1e-12
        assert _angle_to_internal(math.radians(109.5), "radian") == math.radians(109.5)

    def test_writer_inverts_to_output_unit(self):
        """Reader(unit) -> internal radians -> writer(unit) round-trips."""
        from molpy.io.forcefield.xml import _angle_from_internal, _angle_to_internal

        for unit in ("radian", "degree"):
            internal = _angle_to_internal(1.91 if unit == "radian" else 109.5, unit)
            back = _angle_from_internal(internal, unit)
            assert abs(back - (1.91 if unit == "radian" else 109.5)) < 1e-9

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="angle_unit"):
            XMLForceFieldReader(get_forcefield_path("oplsaa.xml"), angle_unit="grad")


class TestAngleUnitDetection:
    """Warnings/errors that flag a likely angle-unit mismatch."""

    def test_degrees_read_as_radians_warns(self):
        """The classic bug: a degree value (104.52) declared radian -> warn."""
        from molpy.io.forcefield.xml import AngleUnitWarning, _normalize_angle

        with pytest.warns(AngleUnitWarning):
            _normalize_angle(104.52, "radian", kind="equilibrium", label="theta0")

    def test_plausible_value_does_not_warn(self):
        """A genuine radian equilibrium angle is silent."""
        import warnings

        from molpy.io.forcefield.xml import _normalize_angle

        with warnings.catch_warnings():
            warnings.simplefilter("error")  # any warning becomes a failure
            rad = _normalize_angle(
                math.radians(109.5), "radian", kind="equilibrium", label="theta0"
            )
        assert abs(rad - math.radians(109.5)) < 1e-9

    def test_phase_out_of_range_warns(self):
        from molpy.io.forcefield.xml import AngleUnitWarning, _normalize_angle

        with pytest.warns(AngleUnitWarning):
            _normalize_angle(400.0, "degree", kind="phase", label="phase1")

    def test_phase_radian_converts_without_warning(self):
        import warnings

        from molpy.io.forcefield.xml import _normalize_angle

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rad = _normalize_angle(
                math.radians(180.0), "radian", kind="phase", label="phase1"
            )
        assert abs(rad - math.pi) < 1e-9


class TestAbsentChargeAttribute:
    """A `charge` the file never states must not become an explicit ``0.0``.

    TIP3P takes charge from the residue (`UseAttributeFromResidue`), so its
    `<Atom>` entries under `NonbondedForce` carry sigma and epsilon and nothing
    else. Recording a fabricated `charge=0.0` there made
    `ForceFieldParams.assign` overwrite the charges already on the graph.
    """

    def test_nonbonded_type_has_no_charge_when_the_file_states_none(self):
        ff = read_xml_forcefield(get_forcefield_path("tip3p.xml"))
        pairstyle = next(iter(ff.get_styles("pair")))
        for typ in pairstyle.types:
            assert "charge" not in typ.params.kwargs, (
                f"{typ.name} invented a charge the force-field file never gave"
            )

    def test_assign_leaves_the_graphs_own_charges_alone(self):
        import molpy as mp
        from molpy.typifier import ForceFieldParams

        ff = read_xml_forcefield(get_forcefield_path("tip3p.xml"))
        water = mp.Atomistic()
        o = water.def_atom(
            element="O", type="tip3p-O", x=0.0, y=0.0, z=0.0, charge=-0.834
        )
        h1 = water.def_atom(
            element="H", type="tip3p-H", x=0.9572, y=0.0, z=0.0, charge=0.417
        )
        h2 = water.def_atom(
            element="H", type="tip3p-H", x=-0.24, y=0.927, z=0.0, charge=0.417
        )
        water.def_bond(o, h1)
        water.def_bond(o, h2)

        typed = ForceFieldParams(ff).assign(water.get_topo(gen_angle=True))

        assert [a["charge"] for a in typed.atoms] == [-0.834, 0.417, 0.417]
        # File states epsilon in kJ/mol (OpenMM); molrs OPLS path converts to
        # kcal/mol for the real-unit force field surface (÷ 4.184).
        assert typed.atoms[0]["epsilon"] == pytest.approx(0.635968 / 4.184)
