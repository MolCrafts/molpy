"""GAFF (Generalized Amber Force Field) typifier implementation."""

import xml.etree.ElementTree as ET
from pathlib import Path

from molpy.core.atomistic import Atomistic
from molpy.core.forcefield import AtomType, ForceField

from .atomistic import (
    ForceFieldAtomisticTypifier,
    ForceFieldAtomTypifier,
)


class GaffAtomTypifier(ForceFieldAtomTypifier):
    """Assign atom types using SMARTS matcher for GAFF force field.

    Key differences from OPLS:
    - Last-match-wins priority (later rules in the file override earlier ones)
    - No type references (no %opls_XXX patterns)
    - Multiple SMARTS rules can map to the same type name
    """

    def _extract_patterns(self) -> dict:
        """Extract SMARTS patterns from GAFF forcefield with position-based priority.

        In GAFF, later rules override earlier rules for the same atom. Priority is
        assigned by order of appearance in the XML: later definitions get higher
        priority. Each SMARTS definition gets a unique key to support multiple
        patterns per type name.

        Reads directly from the XML file to preserve all definitions (including
        duplicate type names with different SMARTS patterns).
        """
        from molpy.parser.smarts import SmartsParser

        from .graph import SMARTSGraph

        pattern_dict = {}
        parser = SmartsParser()

        # Read all atom type definitions directly from the XML file
        # to preserve ordering and duplicates
        definitions = self._read_all_definitions()

        for idx, (type_name, smarts_str) in enumerate(definitions):
            if smarts_str:
                try:
                    key = f"{type_name}_{idx}"
                    pattern = SMARTSGraph(
                        smarts_string=smarts_str,
                        parser=parser,
                        atomtype_name=type_name,
                        priority=idx,
                        source=f"gaff:{type_name}",
                        overrides=set(),
                        target_vertices=[0],
                    )
                    pattern_dict[key] = pattern
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to parse SMARTS for {type_name}: {smarts_str}, error: {e}",
                        stacklevel=2,
                    )

        return pattern_dict

    def _read_all_definitions(self) -> list[tuple[str, str]]:
        """Read all atom type definitions from the GAFF XML file.

        Returns list of (type_name, smarts_string) tuples in file order,
        preserving duplicates.
        """
        # Find the XML file path from the forcefield reader
        # Try to find the XML file via the data module
        try:
            from molpy.data import get_forcefield_path

            xml_path = get_forcefield_path("gaff.xml")
        except (ImportError, FileNotFoundError):
            # Fallback: check common locations
            candidates = [
                Path("src/molpy/data/forcefield/gaff.xml"),
                Path(__file__).parent.parent / "data" / "forcefield" / "gaff.xml",
            ]
            xml_path = None
            for p in candidates:
                if p.exists():
                    xml_path = str(p)
                    break

        if xml_path is None:
            # Fall back to extracting from the forcefield object
            definitions = []
            for at in self.ff.get_types(AtomType):
                smarts_str = at.params.kwargs.get("def_")
                if smarts_str:
                    definitions.append((at.name, smarts_str))
            return definitions

        # Parse XML directly
        tree = ET.parse(xml_path)
        root = tree.getroot()

        definitions = []
        for child in root:
            if child.tag == "AtomTypes":
                for type_elem in child:
                    if type_elem.tag == "Type":
                        name = type_elem.get("name", "")
                        def_str = type_elem.get("def", "")
                        if name and def_str:
                            definitions.append((name, def_str))
                break

        return definitions


class GaffAtomisticTypifier(ForceFieldAtomisticTypifier):
    """GAFF atomistic typifier orchestrator.

    Runs the full typing pipeline:
    atom typing -> pair typing -> bond typing -> angle typing -> dihedral typing
    """

    def _init_typifiers(self) -> None:
        if not self.skip_atom_typing:
            self.atom_typifier = GaffAtomTypifier(self.ff, strict=self.strict_typing)
        super()._init_typifiers()


# Backward-compatible alias
GaffTypifier = GaffAtomisticTypifier
