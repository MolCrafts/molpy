#!/usr/bin/env python3
"""Unit tests for XML force field reader.

This module contains comprehensive tests for:
- XML force field file reading functionality
- Force field parameter extraction and validation
- Atom types, bond types, angle types, dihedral types, and pair types
- Error handling and edge cases

Uses pytest framework with modern Python 3.10+ type hints and Google-style docstrings.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from molpy import AtomisticForcefield, AtomType, BondType, PairType
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

            # Validate parsed atom types
            atomtypes = ff.get_atomtypes()
            assert len(atomtypes) == len(
                expected_atomtypes
            ), f"Expected {len(expected_atomtypes)} atom types, got {len(atomtypes)}"

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
            assert len(pairtypes) >= len(
                expected_pairs
            ), f"Expected at least {len(expected_pairs)} pair types, got {len(pairtypes)}"

            for type_name, expected in expected_pairs.items():
                # Find pair type by atom type name
                found = None
                for pt in pairtypes:
                    if (
                        pt.itom.name == type_name
                        or pt.itom.params.kwargs.get("type_") == type_name
                    ):
                        found = pt
                        break

                assert found is not None, f"Pair type for '{type_name}' not found"
                if expected["charge"] is not None:
                    assert (
                        abs(found.params.kwargs.get("charge", 0) - expected["charge"])
                        < 1e-6
                    )
                if expected["sigma"] is not None:
                    assert (
                        abs(found.params.kwargs.get("sigma", 0) - expected["sigma"])
                        < 1e-6
                    )
                if expected["epsilon"] is not None:
                    assert (
                        abs(found.params.kwargs.get("epsilon", 0) - expected["epsilon"])
                        < 1e-6
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
            bondtypes = ff.get_bondtypes()
            assert len(bondtypes) == len(
                expected_bonds
            ), f"Expected {len(expected_bonds)} bond types, got {len(bondtypes)}"

            for expected in expected_bonds:
                # Find bond type by atom types
                found = None
                for bt in bondtypes:
                    at1_match = (
                        bt.itom.name == expected["type1"]
                        or bt.itom.params.kwargs.get("type_") == expected["type1"]
                        or bt.itom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_match = (
                        bt.jtom.name == expected["type2"]
                        or bt.jtom.params.kwargs.get("type_") == expected["type2"]
                        or bt.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    if at1_match and at2_match:
                        found = bt
                        break

                assert (
                    found is not None
                ), f"Bond type '{expected['type1']}-{expected['type2']}' not found"
                if expected["length"] is not None:
                    assert (
                        abs(found.params.kwargs.get("r0", 0) - expected["length"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    assert abs(found.params.kwargs.get("k", 0) - expected["k"]) < 1e-6

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
                        "angle": float(angle) if angle else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed angle types
            angletypes = ff.get_angletypes()
            assert len(angletypes) == len(
                expected_angles
            ), f"Expected {len(expected_angles)} angle types, got {len(angletypes)}"

            # Track which angles we've matched
            matched_indices = set()
            for expected in expected_angles:
                # Find angle type by atom types (try both forward and reverse)
                found = None
                found_idx = None
                for idx, at in enumerate(angletypes):
                    if idx in matched_indices:
                        continue
                    # Forward match
                    at1_match = (
                        at.itom.name == expected["type1"]
                        or at.itom.params.kwargs.get("type_") == expected["type1"]
                        or at.itom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_match = (
                        at.jtom.name == expected["type2"]
                        or at.jtom.params.kwargs.get("type_") == expected["type2"]
                        or at.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    at3_match = (
                        at.ktom.name == expected["type3"]
                        or at.ktom.params.kwargs.get("type_") == expected["type3"]
                        or at.ktom.params.kwargs.get("class_") == expected["class3"]
                    )
                    # Reverse match
                    at1_rev_match = (
                        at.ktom.name == expected["type1"]
                        or at.ktom.params.kwargs.get("type_") == expected["type1"]
                        or at.ktom.params.kwargs.get("class_") == expected["class1"]
                    )
                    at2_rev_match = (
                        at.jtom.name == expected["type2"]
                        or at.jtom.params.kwargs.get("type_") == expected["type2"]
                        or at.jtom.params.kwargs.get("class_") == expected["class2"]
                    )
                    at3_rev_match = (
                        at.itom.name == expected["type3"]
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

                assert (
                    found is not None
                ), f"Angle type '{expected['type1']}-{expected['type2']}-{expected['type3']}' not found"
                if found_idx is not None:
                    matched_indices.add(found_idx)
                if expected["angle"] is not None:
                    assert (
                        abs(found.params.kwargs.get("theta0", 0) - expected["angle"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    assert abs(found.params.kwargs.get("k", 0) - expected["k"]) < 1e-6

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
            atomtypes = ff.get_atomtypes()
            assert len(atomtypes) == len(
                expected_atomtypes
            ), f"Expected {len(expected_atomtypes)} atom types, got {len(atomtypes)}"

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
            bondtypes = ff.get_bondtypes()
            assert len(bondtypes) == len(
                expected_bonds
            ), f"Expected {len(expected_bonds)} bond types, got {len(bondtypes)}"

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

                assert (
                    found is not None
                ), f"Bond type '{expected['type1']}-{expected['type2']}' not found"
                if expected["length"] is not None:
                    assert (
                        abs(found.params.kwargs.get("r0", 0) - expected["length"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    assert abs(found.params.kwargs.get("k", 0) - expected["k"]) < 1e-6

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
                        "angle": float(angle) if angle else None,
                        "k": float(k) if k else None,
                    }
                )

            # Validate parsed angle types
            angletypes = ff.get_angletypes()
            assert len(angletypes) == len(
                expected_angles
            ), f"Expected {len(expected_angles)} angle types, got {len(angletypes)}"

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

                assert (
                    found is not None
                ), f"Angle type '{expected['type1']}-{expected['type2']}-{expected['type3']}' not found"
                if expected["angle"] is not None:
                    assert (
                        abs(found.params.kwargs.get("theta0", 0) - expected["angle"])
                        < 1e-6
                    )
                if expected["k"] is not None:
                    assert abs(found.params.kwargs.get("k", 0) - expected["k"]) < 1e-6

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
            assert len(pairtypes) >= len(
                expected_pairs
            ), f"Expected at least {len(expected_pairs)} pair types, got {len(pairtypes)}"

            for type_name, expected in expected_pairs.items():
                # Find pair type by atom type name
                found = None
                for pt in pairtypes:
                    if (
                        pt.itom.name == type_name
                        or pt.itom.params.kwargs.get("type_") == type_name
                    ):
                        found = pt
                        break

                assert found is not None, f"Pair type for '{type_name}' not found"
                if expected["sigma"] is not None:
                    assert (
                        abs(found.params.kwargs.get("sigma", 0) - expected["sigma"])
                        < 1e-6
                    )
                if expected["epsilon"] is not None:
                    assert (
                        abs(found.params.kwargs.get("epsilon", 0) - expected["epsilon"])
                        < 1e-6
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
            atomtypes = ff.get_atomtypes()
            # The parser may create additional types, so we check >=
            assert (
                len(atomtypes) >= expected_count
            ), f"Expected at least {expected_count} atom types, got {len(atomtypes)}"

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
        assert isinstance(ff, AtomisticForcefield)
        assert len(ff.get_atomtypes()) > 0

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
        existing_ff = AtomisticForcefield(name="test", units="real")

        # Read XML into existing force field
        ff = read_xml_forcefield(xml_file, forcefield=existing_ff)

        # Should be the same object
        assert ff is existing_ff
        assert len(ff.get_atomtypes()) > 0

    def test_parse_bond_with_class_only_creates_wildcard_atomtypes(
        self, TEST_DATA_DIR: Path
    ) -> None:
        """Test that parsing bonds with only class attributes creates wildcard AtomTypes.

        When a Bond element has class1/class2 but no type1/type2, the parser should
        create wildcard AtomTypes (type="*", class=class_name) for both classes.
        """
        xml_file = TEST_DATA_DIR / "xml" / "oplsaa.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        ff = read_xml_forcefield(xml_file)

        # Check that O_3 wildcard AtomType exists
        o3_wildcard = None
        for at in ff.get_types(AtomType):
            at_class = at.params.kwargs.get("class_", "")
            at_type = at.params.kwargs.get("type_", "")
            if at_class == "O_3" and at_type == "*":
                o3_wildcard = at
                break

        assert (
            o3_wildcard is not None
        ), "O_3 wildcard AtomType should be created when parsing bonds"
        assert o3_wildcard.name == "O_3", "O_3 wildcard AtomType should have name 'O_3'"

        # Check that C wildcard AtomType exists
        c_wildcard = None
        for at in ff.get_types(AtomType):
            at_class = at.params.kwargs.get("class_", "")
            at_type = at.params.kwargs.get("type_", "")
            if at_class == "C" and at_type == "*":
                c_wildcard = at
                break

        assert c_wildcard is not None, "C wildcard AtomType should exist"

        # Check that C - O_3 bond type exists
        bond_found = False
        for bt in ff.get_types(BondType):
            at1_class = (
                bt.itom.params.kwargs.get("class_", "")
                if hasattr(bt.itom, "params")
                else ""
            )
            at2_class = (
                bt.jtom.params.kwargs.get("class_", "")
                if hasattr(bt.jtom, "params")
                else ""
            )
            if (at1_class == "C" and at2_class == "O_3") or (
                at1_class == "O_3" and at2_class == "C"
            ):
                bond_found = True
                # Verify it uses the wildcard AtomTypes
                assert (bt.itom is c_wildcard and bt.jtom is o3_wildcard) or (
                    bt.itom is o3_wildcard and bt.jtom is c_wildcard
                ), "Bond should use wildcard AtomTypes"
                break

        assert bond_found, "C - O_3 bond type should exist and use wildcard AtomTypes"

    def test_class_based_bond_typing_works(self, TEST_DATA_DIR: Path) -> None:
        """Test that class-based bond typing works after parsing.

        This test verifies that bonds can be typed using class matching,
        which requires wildcard AtomTypes to be created during parsing.
        """
        from molpy import Atom, Atomistic, Bond
        from molpy.typifier.atomistic import OplsBondTypifier

        xml_file = TEST_DATA_DIR / "xml" / "oplsaa.xml"
        assert xml_file.exists(), f"Test file not found: {xml_file}"

        ff = read_xml_forcefield(xml_file)

        # Create a simple structure with opls_269 and opls_267 atoms
        asm = Atomistic()
        atom1 = Atom(symbol="O")
        atom1.data["type"] = "opls_269"  # class="O_3"
        atom2 = Atom(symbol="C")
        atom2.data["type"] = "opls_267"  # class="C"
        asm.add_entity(atom1, atom2)

        bond = Bond(atom1, atom2)
        asm.add_link(bond)

        # Typify bond
        typifier = OplsBondTypifier(ff)
        typifier.typify(bond)

        # Should have a type assigned
        assert bond.data.get("type") is not None, "Bond should have a type assigned"
        assert "k" in bond.data or "r0" in bond.data, "Bond should have parameters"
