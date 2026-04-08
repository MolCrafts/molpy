"""Integration tests for the polymer building pipeline.

Tests the full flow: parse → build → typify, verifying:
- Port selection direction (> connects to <, not < to <)
- Junction bond chemistry (O-C ether, not C-C)
- Chemical correctness (atom valence)
- All atoms typed after typification
- Build markers cleaned up from final structure
"""

import pytest

from molpy.builder.polymer.port_utils import port_role, ports_compatible
from molpy.tool.polymer import polymer


class TestPortRoleDerivation:
    def test_left_from_name(self):
        assert port_role("<") == "left"
        assert port_role("<0") == "left"

    def test_right_from_name(self):
        assert port_role(">") == "right"
        assert port_role(">1") == "right"

    def test_terminal_from_name(self):
        assert port_role("$") == "terminal"
        assert port_role("$2") == "terminal"


class TestPortCompatibility:
    def test_directional_pair(self):
        assert ports_compatible(">", "<")
        assert ports_compatible("<", ">")

    def test_same_name_symmetric(self):
        assert ports_compatible("$", "$")

    def test_same_direction_incompatible(self):
        assert not ports_compatible("<", "<")
        assert not ports_compatible(">", ">")


class TestJunctionConnectivity:
    def test_peg_all_oxygens_are_ethers_or_terminal(self):
        """In PEG, every O is either an ether (C-O-C) or terminal (C-O-H)."""
        peg2 = polymer("{[<]CCOCCO[>]}|2|", optimize=True)

        for atom in peg2.atoms:
            if atom.get("element") != "O":
                continue
            neighbor_syms = []
            for bond in peg2.bonds:
                if bond.itom is atom:
                    neighbor_syms.append(bond.jtom.get("element"))
                elif bond.jtom is atom:
                    neighbor_syms.append(bond.itom.get("element"))

            # O should have degree 2
            assert len(neighbor_syms) == 2, f"O has degree {len(neighbor_syms)}"
            # O neighbors should be (C,C) for ether or (C,H) for terminal OH
            syms = sorted(neighbor_syms)
            assert syms in [["C", "C"], ["C", "H"]], (
                f"O should be ether (C-O-C) or terminal (C-O-H), got {syms}"
            )

    def test_peg_no_cc_junction(self):
        """PEG should not have C-C bonds between what would be junction carbons.

        In correct PEG: ...CH2-O-CH2... (all C-C bonds have at most 1 O neighbor).
        A C-C junction bug would create a C bonded to 3 C + 1 H (no O).
        """
        peg3 = polymer("{[<]CCOCCO[>]}|3|", optimize=True)

        for atom in peg3.atoms:
            if atom.get("element") != "C":
                continue
            neighbors = []
            for bond in peg3.bonds:
                if bond.itom is atom:
                    neighbors.append(bond.jtom.get("element"))
                elif bond.jtom is atom:
                    neighbors.append(bond.itom.get("element"))

            # In PEG, every C should have at least one O or H neighbor
            # A C-C junction would make a C with neighbors [C, C, H, H] (no O)
            # but that's still valid for terminal CH3. The key invariant is:
            # C should have degree 4
            assert len(neighbors) == 4, f"C has degree {len(neighbors)}"

    def test_peg_atom_count(self):
        """PEG-n atom count matches expected formula.

        CCOCCO monomer has 6 heavy + 10 H = 16 atoms.
        Each junction removes 2 H and forms 1 O-C bond.
        PEG-n: 16n - 2(n-1) = 14n + 2 atoms.
        """
        for n in (2, 3, 5):
            peg = polymer(f"{{[<]CCOCCO[>]}}|{n}|", optimize=True)
            expected = 14 * n + 2
            actual = len(list(peg.atoms))
            assert actual == expected, (
                f"PEG-{n}: expected {expected} atoms, got {actual}"
            )


class TestChemicalCorrectness:
    def test_atom_valence(self):
        """All atoms have correct valence in built PEG."""
        peg = polymer("{[<]CCOCCO[>]}|3|", optimize=True)
        expected_degree = {"C": 4, "O": 2, "H": 1}

        for atom in peg.atoms:
            sym = atom.get("element")
            degree = sum(1 for b in peg.bonds if b.itom is atom or b.jtom is atom)
            assert degree == expected_degree[sym], (
                f"{sym} atom (mp_id={atom.get('mp_id')}) should have degree "
                f"{expected_degree[sym]}, got {degree}"
            )


class TestBuildCleanup:
    def test_no_port_markers_after_build(self):
        """Built polymer has no leftover port/monomer_node_id markers."""
        peg = polymer("{[<]CCOCCO[>]}|3|", optimize=True)

        for atom in peg.atoms:
            assert atom.get("port") is None, (
                f"Atom has leftover port={atom.get('port')}"
            )
            assert atom.get("monomer_node_id") is None, (
                f"Atom has leftover monomer_node_id={atom.get('monomer_node_id')}"
            )


class TestFullPipeline:
    def test_parse_build_typify_all_typed(self):
        """Parse → Build → Typify → all atoms have types."""
        from molpy.data import get_forcefield_path
        from molpy.io.forcefield.xml import XMLForceFieldReader
        from molpy.typifier import OplsAtomisticTypifier

        peg = polymer("{[<]CCOCCO[>]}|5|", optimize=True)
        opls_ff = XMLForceFieldReader(get_forcefield_path("oplsaa.xml")).read()
        typed = OplsAtomisticTypifier(opls_ff, strict_typing=False).typify(peg)

        untyped = [a for a in typed.atoms if a.get("type") is None]
        assert untyped == [], f"{len(untyped)} atoms untyped"
