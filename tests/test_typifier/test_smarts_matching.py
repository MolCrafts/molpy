"""Comprehensive SMARTS matching tests inspired by foyer.

Tests cover:
- SMARTS pattern matching against real molecular graphs
- Ring detection (uniqueness, fused rings, ring count)
- Operator precedence (AND, OR, NOT, weak AND)
- Negation matching
- Bond order matching
- Aromatic matching
- Immutability guarantees
- Edge cases

References:
    foyer project: https://github.com/mosdef-hub/foyer
    SMARTS specification: https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html
"""

import pytest

from molpy import Atom, Atomistic, Bond
from molpy.parser.smarts import SmartsParser
from molpy.typifier.adapter import build_mol_graph
from molpy.typifier.graph import SMARTSGraph


# ===================================================================
# Test Helpers
# ===================================================================


def _make_pattern(smarts: str, name: str = "test") -> SMARTSGraph:
    """Create a SMARTSGraph pattern from a SMARTS string."""
    parser = SmartsParser()
    return SMARTSGraph(
        smarts_string=smarts,
        parser=parser,
        atomtype_name=name,
        priority=0,
        target_vertices=[0],
    )


def _count_matches(smarts: str, mol_graph) -> int:
    """Count how many atoms in mol_graph match the SMARTS pattern."""
    pattern = _make_pattern(smarts)
    matches = pattern.find_matches(mol_graph)
    return len(matches)


def _build_linear_chain(n: int, symbol: str = "C") -> Atomistic:
    """Build a linear chain of n atoms."""
    asm = Atomistic()
    atoms = [Atom(element=symbol) for _ in range(n)]
    for a in atoms:
        asm.add_entity(a)
    for i in range(n - 1):
        asm.add_link(Bond(atoms[i], atoms[i + 1]))
    return asm


def _build_ring(n: int, symbol: str = "C") -> Atomistic:
    """Build a ring of n atoms."""
    asm = Atomistic()
    atoms = [Atom(element=symbol) for _ in range(n)]
    for a in atoms:
        asm.add_entity(a)
    for i in range(n):
        asm.add_link(Bond(atoms[i], atoms[(i + 1) % n]))
    return asm


def _build_ethane() -> Atomistic:
    """Build ethane (C2H6)."""
    asm = Atomistic()
    c1 = Atom(element="C")
    c2 = Atom(element="C")
    hs = [Atom(element="H") for _ in range(6)]
    asm.add_entity(c1, c2, *hs)
    asm.add_link(Bond(c1, c2))
    # 3 H on C1
    for h in hs[:3]:
        asm.add_link(Bond(c1, h))
    # 3 H on C2
    for h in hs[3:]:
        asm.add_link(Bond(c2, h))
    return asm


def _build_propene() -> Atomistic:
    """Build propene (CH2=CH-CH3) with explicit hydrogens."""
    asm = Atomistic()
    c1 = Atom(element="C")  # =CH2
    c2 = Atom(element="C")  # =CH-
    c3 = Atom(element="C")  # -CH3

    h1 = Atom(element="H")  # on C1
    h2 = Atom(element="H")  # on C1
    h3 = Atom(element="H")  # on C2
    h4 = Atom(element="H")  # on C3
    h5 = Atom(element="H")  # on C3
    h6 = Atom(element="H")  # on C3

    asm.add_entity(c1, c2, c3, h1, h2, h3, h4, h5, h6)

    # C=C double bond
    double_bond = Bond(c1, c2)
    double_bond.data["order"] = 2
    asm.add_link(double_bond)

    # C-C single bond
    asm.add_link(Bond(c2, c3))

    # C1-H
    asm.add_link(Bond(c1, h1))
    asm.add_link(Bond(c1, h2))
    # C2-H
    asm.add_link(Bond(c2, h3))
    # C3-H
    asm.add_link(Bond(c3, h4))
    asm.add_link(Bond(c3, h5))
    asm.add_link(Bond(c3, h6))

    return asm


def _build_benzene() -> Atomistic:
    """Build benzene with aromatic flags."""
    asm = Atomistic()
    carbons = [Atom(element="C", is_aromatic=True) for _ in range(6)]
    hydrogens = [Atom(element="H") for _ in range(6)]

    for c, h in zip(carbons, hydrogens):
        asm.add_entity(c, h)

    for i in range(6):
        bond = Bond(carbons[i], carbons[(i + 1) % 6])
        bond.data["order"] = 1.5
        bond.data["kind"] = ":"
        asm.add_link(bond)

    for c, h in zip(carbons, hydrogens):
        asm.add_link(Bond(c, h))

    return asm


def _build_naphthalene() -> Atomistic:
    """Build naphthalene (fused bicyclic ring)."""
    asm = Atomistic()
    # 10 carbons: c0-c9
    carbons = [Atom(element="C", is_aromatic=True) for _ in range(10)]
    # 8 hydrogens
    hydrogens = [Atom(element="H") for _ in range(8)]

    for c in carbons:
        asm.add_entity(c)
    for h in hydrogens:
        asm.add_entity(h)

    # Ring 1: 0-1-2-3-4-5
    # Ring 2: 5-6-7-8-9-4
    ring_bonds = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 0),  # ring 1
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 4),  # ring 2
    ]
    for i, j in ring_bonds:
        bond = Bond(carbons[i], carbons[j])
        bond.data["order"] = 1.5
        bond.data["kind"] = ":"
        asm.add_link(bond)

    # H on non-junction carbons (0,1,2,3,6,7,8,9)
    h_carbons = [0, 1, 2, 3, 6, 7, 8, 9]
    for hi, ci in enumerate(h_carbons):
        asm.add_link(Bond(carbons[ci], hydrogens[hi]))

    return asm


def _build_cyclohexane() -> Atomistic:
    """Build cyclohexane (C6H12)."""
    asm = Atomistic()
    carbons = [Atom(element="C") for _ in range(6)]
    hydrogens = [Atom(element="H") for _ in range(12)]

    for c in carbons:
        asm.add_entity(c)
    for h in hydrogens:
        asm.add_entity(h)

    # Ring bonds
    for i in range(6):
        asm.add_link(Bond(carbons[i], carbons[(i + 1) % 6]))

    # H bonds: 2 per carbon
    for i in range(6):
        asm.add_link(Bond(carbons[i], hydrogens[2 * i]))
        asm.add_link(Bond(carbons[i], hydrogens[2 * i + 1]))

    return asm


def _build_water() -> Atomistic:
    """Build water (H2O)."""
    asm = Atomistic()
    o = Atom(element="O")
    h1 = Atom(element="H")
    h2 = Atom(element="H")
    asm.add_entity(o, h1, h2)
    asm.add_link(Bond(o, h1))
    asm.add_link(Bond(o, h2))
    return asm


def _build_methanol() -> Atomistic:
    """Build methanol (CH3OH)."""
    asm = Atomistic()
    c = Atom(element="C")
    o = Atom(element="O")
    h1 = Atom(element="H")
    h2 = Atom(element="H")
    h3 = Atom(element="H")
    h4 = Atom(element="H")  # hydroxyl H
    asm.add_entity(c, o, h1, h2, h3, h4)
    asm.add_link(Bond(c, o))
    asm.add_link(Bond(c, h1))
    asm.add_link(Bond(c, h2))
    asm.add_link(Bond(c, h3))
    asm.add_link(Bond(o, h4))
    return asm


# ===================================================================
# Tests: Element and Atomic Number Matching
# ===================================================================


class TestElementMatching:
    """Test basic element symbol and atomic number matching."""

    def test_match_carbon_by_symbol(self):
        """[C] matches carbon atoms."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#6]", g) == 2

    def test_match_hydrogen_by_symbol(self):
        """[H] matches hydrogen atoms."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#1]", g) == 6

    def test_wildcard_matches_all(self):
        """[*] matches all atoms."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[*]", g) == 8

    def test_no_match_wrong_element(self):
        """[#7] (nitrogen) should not match in ethane."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#7]", g) == 0


# ===================================================================
# Tests: Neighbor Count (Connectivity)
# ===================================================================


class TestNeighborCount:
    """Test X (neighbor count) matching."""

    def test_carbon_x4(self):
        """[#6X4] matches sp3 carbon (4 neighbors)."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        # Both carbons have 4 neighbors (1 C + 3 H)
        assert _count_matches("[#6X4]", g) == 2

    def test_hydrogen_x1(self):
        """[#1X1] matches terminal hydrogen."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#1X1]", g) == 6

    def test_carbon_x3_propene(self):
        """[#6X3] matches sp2 carbons in propene."""
        asm = _build_propene()
        g, *_ = build_mol_graph(asm)
        # C1 (=CH2) has 3 neighbors, C2 (=CH-) has 3 neighbors
        assert _count_matches("[#6X3]", g) == 2

    def test_carbon_x4_propene(self):
        """[#6X4] matches sp3 carbon in propene."""
        asm = _build_propene()
        g, *_ = build_mol_graph(asm)
        # C3 (-CH3) has 4 neighbors
        assert _count_matches("[#6X4]", g) == 1

    def test_oxygen_x2(self):
        """[#8X2] matches oxygen with 2 neighbors (water, alcohol)."""
        asm = _build_water()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#8X2]", g) == 1


# ===================================================================
# Tests: Ring Detection
# ===================================================================


class TestRingDetection:
    """Test ring-related SMARTS matching.

    Adapted from foyer test_smarts.py: test_ringness, test_uniqueness,
    test_fused_ring, test_ring_count.
    """

    def test_ring_pattern_matches_ring(self):
        """Ring SMARTS matches ring structure."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)
        # [#6;R] matches carbon in any ring
        assert _count_matches("[#6;R]", g) == 6

    def test_ring_pattern_does_not_match_chain(self):
        """Ring SMARTS does not match linear chain."""
        asm = _build_linear_chain(6)
        g, *_ = build_mol_graph(asm)
        # [#6;R] should not match any carbon in a chain
        assert _count_matches("[#6;R]", g) == 0

    def test_ring_size_six(self):
        """[#6;r6] matches atoms in 6-membered ring."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#6;r6]", g) == 6

    def test_ring_size_five_no_match(self):
        """[#6;r5] should not match in cyclohexane (no 5-ring)."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#6;r5]", g) == 0

    def test_ring_uniqueness(self):
        """Ring size patterns match only the correct ring size.

        Adapted from foyer test_uniqueness: a 4-membered ring should only
        be matched by [r4], not [r5] or [r6].
        """
        asm = _build_ring(4)
        g, *_ = build_mol_graph(asm)

        assert _count_matches("[#6;r4]", g) == 4
        assert _count_matches("[#6;r5]", g) == 0
        assert _count_matches("[#6;r6]", g) == 0

    def test_ring_count_single_ring(self):
        """[#6;R1] matches atoms in exactly one ring."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#6;R1]", g) == 6

    def test_fused_ring_r2(self):
        """[R2] matches junction atoms in fused rings.

        Adapted from foyer test_ring_count: In naphthalene, atoms 4 and 5
        are in 2 rings (junction atoms).
        """
        asm = _build_naphthalene()
        g, *_ = build_mol_graph(asm)

        # Junction atoms should be in 2 rings
        r2_count = _count_matches("[#6;R2]", g)
        assert r2_count == 2, f"Expected 2 junction atoms, got {r2_count}"

        # Non-junction atoms in exactly 1 ring
        r1_count = _count_matches("[#6;R1]", g)
        assert r1_count == 8, f"Expected 8 non-junction atoms, got {r1_count}"


# ===================================================================
# Tests: Aromatic Matching
# ===================================================================


class TestAromaticMatching:
    """Test aromatic atom and bond matching."""

    def test_aromatic_carbon(self):
        """Lowercase [c] matches aromatic carbon."""
        asm = _build_benzene()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[c]", g) == 6

    def test_aromatic_property(self):
        """[a] matches any aromatic atom."""
        asm = _build_benzene()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[a]", g) == 6

    def test_aliphatic_property(self):
        """[A] matches any aliphatic atom."""
        asm = _build_benzene()
        g, *_ = build_mol_graph(asm)
        # Only H atoms are aliphatic in benzene
        assert _count_matches("[A]", g) == 6  # 6 hydrogens

    def test_no_aromatic_in_ethane(self):
        """[c] should not match in ethane."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[c]", g) == 0


# ===================================================================
# Tests: Hydrogen Count
# ===================================================================


class TestHydrogenCount:
    """Test hydrogen count matching."""

    def test_hydrogen_count_methanol(self):
        """[#6H3] matches carbon with 3 hydrogens."""
        asm = _build_methanol()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#6;H3]", g) == 1  # methyl carbon

    def test_oxygen_h1(self):
        """[#8;H1] matches oxygen with 1 hydrogen (hydroxyl)."""
        asm = _build_methanol()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[#8;H1]", g) == 1


# ===================================================================
# Tests: Bond Order Matching
# ===================================================================


class TestBondOrderMatching:
    """Test bond type matching in SMARTS.

    Adapted from foyer test_bond_order.
    """

    def test_double_bond(self):
        """[C](=C) matches carbon with double bond to carbon."""
        asm = _build_propene()
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6](=[#6])")
        matches = pattern.find_matches(g)
        # Both C1 and C2 participate in the double bond
        assert len(matches) == 2

    def test_single_bond_only(self):
        """[C](-C) matches carbon with single bond to carbon."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6](-[#6])")
        matches = pattern.find_matches(g)
        assert len(matches) == 2  # Both carbons have single bond to each other


# ===================================================================
# Tests: Negation
# ===================================================================


class TestNegation:
    """Test NOT (!) operator in SMARTS.

    Adapted from foyer test_not.
    """

    def test_not_oxygen(self):
        """[!#8] matches everything except oxygen."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        # Ethane has 8 atoms (2C + 6H), no oxygen
        assert _count_matches("[!#8]", g) == 8

    def test_not_carbon(self):
        """[!#6] matches non-carbon atoms."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        # 6 hydrogens are not carbon
        assert _count_matches("[!#6]", g) == 6

    def test_not_nitrogen(self):
        """[!#7] matches everything (no nitrogen in ethane)."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[!#7]", g) == 8

    def test_not_carbon_and_not_hydrogen(self):
        """[!#6;!#1] matches atoms that are neither C nor H."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        # Nothing in ethane is neither C nor H
        assert _count_matches("[!#6;!#1]", g) == 0


# ===================================================================
# Tests: Operator Precedence
# ===================================================================


class TestOperatorPrecedence:
    """Test SMARTS operator precedence: & > , > ;

    Adapted from foyer test_precedence.

    Ethane (C2H6) has 8 atoms: 2 C + 6 H.
    """

    def test_or_weak_and(self):
        """[#6,#8;#6] = (C OR O) weak_AND C = C matches."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        # [#6,#8;#6]: (match C or O) AND (match C) => matches C
        assert _count_matches("[#6,#8;#6]", g) == 2

    def test_and_or(self):
        """[!#6&!#1] = (NOT C) AND (NOT H)."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[!#6&!#1]", g) == 0

    def test_not_c_or_c(self):
        """[!#6,#6] = (NOT C) OR C = everything."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)
        assert _count_matches("[!#6,#6]", g) == 8

    def test_complex_precedence(self):
        """[!#6;#8,#6] = NOT C WEAK_AND (O OR C)."""
        asm = _build_methanol()
        g, *_ = build_mol_graph(asm)
        # NOT carbon: H,O ; then (O or C): so only O matches
        assert _count_matches("[!#6;#8,#6]", g) == 1  # oxygen only


# ===================================================================
# Tests: Ring Bond (@) and Negated Bond (!:)
# ===================================================================


class TestRingAndNegatedBonds:
    """Test @ ring bond and !: negated bond matching."""

    def test_ring_bond_in_cyclohexane(self):
        """Ring bond @ matches bonds within a ring."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)

        # [#6](@[#6]) means: carbon bonded to carbon via ring bond
        pattern = _make_pattern("[#6](@[#6])")
        matches = pattern.find_matches(g)
        assert len(matches) == 6, f"Expected 6 ring-bonded carbons, got {len(matches)}"

    def test_ring_bond_not_in_chain(self):
        """Ring bond @ should not match in linear chain."""
        asm = _build_linear_chain(6)
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6](@[#6])")
        matches = pattern.find_matches(g)
        assert len(matches) == 0

    def test_negated_aromatic_bond(self):
        """[#6](!:[#6]) matches C bonded to C via non-aromatic bond."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6](!:[#6])")
        matches = pattern.find_matches(g)
        assert len(matches) == 2  # Both carbons are bonded non-aromatically


# ===================================================================
# Tests: Immutability
# ===================================================================


class TestImmutability:
    """Test that typifiers do not mutate input objects."""

    def test_gaff_atom_typifier_immutable(self):
        """GaffAtomTypifier.typify() must not mutate input."""
        from molpy.io.forcefield.xml import XMLForceFieldReader
        from molpy.typifier.gaff import GaffAtomTypifier

        ff = XMLForceFieldReader("src/molpy/data/forcefield/gaff.xml").read()
        typifier = GaffAtomTypifier(ff, strict=False)

        asm = _build_ethane()
        original_atom_data = [dict(atom.data) for atom in asm.atoms]

        result = typifier.typify(asm)

        # Original atoms should be untouched
        for atom, orig_data in zip(asm.atoms, original_atom_data):
            assert dict(atom.data) == orig_data, "Original atom data was mutated"

        # Result is a new object
        assert result is not asm

        # Result should have types
        for atom in result.atoms:
            assert atom.get("type") is not None

    def test_opls_bond_typifier_immutable_orchestrator(self):
        """OplsTypifier returns new struct without mutating input."""
        from molpy import AtomisticForcefield

        from molpy.typifier.opls import OplsTypifier

        ff = AtomisticForcefield()
        astyle = ff.def_atomstyle("full")
        at1 = astyle.def_type("CA", type_="CA", class_="CT")
        at2 = astyle.def_type("HA", type_="HA", class_="HC")

        bstyle = ff.def_bondstyle("harmonic")
        bstyle.def_type(at1, at2, k0=1000.0, r0=1.08)

        typifier = OplsTypifier(
            ff,
            skip_atom_typing=True,
            skip_angle_typing=True,
            skip_dihedral_typing=True,
            skip_pair_typing=True,
        )

        asm = Atomistic()
        c = Atom(element="C")
        h = Atom(element="H")
        c.data["type"] = "CA"
        h.data["type"] = "HA"
        asm.add_entity(c, h)
        asm.add_link(Bond(c, h))

        # Capture original state
        original_bond_data = [dict(b.data) for b in asm.bonds]

        result = typifier.typify(asm)

        # Input not mutated
        assert result is not asm
        for bond, orig in zip(asm.bonds, original_bond_data):
            assert dict(bond.data) == orig

    def test_gaff_atomistic_typifier_immutable(self):
        """GaffTypifier returns new struct without mutating input."""
        from molpy.io.forcefield.xml import XMLForceFieldReader
        from molpy.typifier.gaff import GaffTypifier

        ff = XMLForceFieldReader("src/molpy/data/forcefield/gaff.xml").read()
        typifier = GaffTypifier(ff, strict_typing=False)

        asm = _build_ethane()
        original_atom_count = len(list(asm.atoms))
        original_bond_count = len(list(asm.bonds))

        result = typifier.typify(asm)

        # Original unchanged
        assert result is not asm
        assert len(list(asm.atoms)) == original_atom_count
        assert len(list(asm.bonds)) == original_bond_count
        for atom in asm.atoms:
            assert atom.get("type") is None


# ===================================================================
# Tests: End-to-End GAFF Typing
# ===================================================================


class TestEndToEndGaffTyping:
    """Test complete GAFF typing workflows on real molecules."""

    @pytest.fixture(scope="class")
    def gaff_ff(self):
        from molpy.io.forcefield.xml import XMLForceFieldReader

        return XMLForceFieldReader("src/molpy/data/forcefield/gaff.xml").read()

    def test_ethane_all_atoms_typed(self, gaff_ff):
        """All atoms in ethane should be typed."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_ethane()
        result = typifier.typify(asm)

        typed_count = sum(1 for a in result.atoms if a.get("type") is not None)
        assert typed_count == 8, f"Expected 8 typed atoms, got {typed_count}"

    def test_ethane_type_counts(self, gaff_ff):
        """Ethane: 2 c3 carbons, 6 hc hydrogens."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_ethane()
        result = typifier.typify(asm)

        types = [a.get("type") for a in result.atoms]
        assert types.count("c3") == 2
        assert types.count("hc") == 6

    def test_benzene_type_counts(self, gaff_ff):
        """Benzene: 6 ca carbons, 6 ha hydrogens."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_benzene()
        result = typifier.typify(asm)

        types = [a.get("type") for a in result.atoms]
        assert types.count("ca") == 6, f"Expected 6 ca, got types: {types}"
        assert types.count("ha") == 6, f"Expected 6 ha, got types: {types}"

    def test_water_types(self, gaff_ff):
        """Water: 1 ow oxygen, 2 hw hydrogens."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_water()
        result = typifier.typify(asm)

        types = [a.get("type") for a in result.atoms]
        o_types = [t for t in types if t and t.startswith("o")]
        h_types = [t for t in types if t and t.startswith("h")]
        assert len(o_types) == 1
        assert len(h_types) == 2

    def test_methanol_types(self, gaff_ff):
        """Methanol: c3, oh, hc*3, ho."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_methanol()
        result = typifier.typify(asm)

        types = [a.get("type") for a in result.atoms]
        assert "c3" in types
        assert "oh" in types
        assert "ho" in types

    def test_full_pipeline_bonds_typed(self, gaff_ff):
        """Full pipeline produces typed bonds."""
        from molpy.typifier.gaff import GaffTypifier

        typifier = GaffTypifier(
            gaff_ff,
            skip_pair_typing=True,
            skip_angle_typing=True,
            skip_dihedral_typing=True,
            strict_typing=False,
        )

        asm = _build_ethane()
        result = typifier.typify(asm)

        typed_bonds = [b for b in result.bonds if b.data.get("type") is not None]
        assert len(typed_bonds) > 0

    def test_propene_sp2_sp3_distinction(self, gaff_ff):
        """Propene correctly distinguishes sp2 (c2) and sp3 (c3) carbons."""
        from molpy.typifier.gaff import GaffAtomTypifier

        typifier = GaffAtomTypifier(gaff_ff, strict=False)

        asm = _build_propene()
        result = typifier.typify(asm)

        c_types = [a.get("type") for a in result.atoms if a.get("element") == "C"]
        # Should have sp2 and sp3 carbons
        assert "c3" in c_types, f"Expected c3 in {c_types}"
        assert any(t in ("c2", "ce", "cf") for t in c_types), (
            f"Expected sp2 carbon type in {c_types}"
        )


# ===================================================================
# Tests: Lazy Cycle Detection
# ===================================================================


class TestLazyCycleDetection:
    """Test that cycle detection is lazy (only when needed).

    Adapted from foyer test_lazy_cycle_finding.
    """

    def test_simple_pattern_no_ring_computation(self):
        """Simple patterns like [C] should not trigger ring computation."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6]")
        # Reset cycles on graph to check if calc_signature adds them
        if "cycles" in g.vs.attributes():
            g.vs["cycles"] = [set() for _ in g.vs]

        pattern.calc_signature(g)

        # Cycles should still be empty (not computed for [#6])
        for v in g.vs:
            cycles = v["cycles"] if "cycles" in v.attributes() else set()
            assert len(cycles) == 0

    def test_ring_pattern_triggers_computation(self):
        """Ring patterns like [#6;R] should trigger ring computation."""
        asm = _build_cyclohexane()
        g, *_ = build_mol_graph(asm)

        # Clear existing cycle data
        g.vs["cycles"] = [set() for _ in g.vs]

        pattern = _make_pattern("[#6;R]")
        pattern.calc_signature(g)

        # Cycles should be computed for carbons in ring
        ring_atoms = [v for v in g.vs if len(v["cycles"]) > 0]
        assert len(ring_atoms) > 0


# ===================================================================
# Tests: Multi-atom SMARTS patterns
# ===================================================================


class TestMultiAtomPatterns:
    """Test multi-atom SMARTS patterns."""

    def test_two_atom_pattern(self):
        """[#6][#6] matches C-C."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)

        pattern = _make_pattern("[#6][#6]")
        matches = pattern.find_matches(g)
        # Both carbons should match as the first atom
        assert len(matches) == 2

    def test_three_atom_pattern_water(self):
        """[#1][#8][#1] matches H-O-H."""
        asm = _build_water()
        g, *_ = build_mol_graph(asm)

        # Create pattern matching full H-O-H
        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[#1][#8][#1]",
            parser=parser,
            atomtype_name="test",
            priority=0,
            target_vertices=[0],
        )
        matches = pattern.find_matches(g)
        assert len(matches) == 2  # Both H can be first atom

    def test_branch_pattern(self):
        """[#6]([#1])([#1])[#1] matches CH3 group."""
        asm = _build_ethane()
        g, *_ = build_mol_graph(asm)

        parser = SmartsParser()
        pattern = SMARTSGraph(
            smarts_string="[#6]([#1])([#1])[#1]",
            parser=parser,
            atomtype_name="test",
            priority=0,
            target_vertices=[0],
        )
        matches = pattern.find_matches(g)
        # Both carbons in ethane have 3 H neighbors
        assert len(matches) == 2
