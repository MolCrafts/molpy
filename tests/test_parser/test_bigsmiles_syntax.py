"""Comprehensive BigSMILES / G-BigSMILES syntax tests.

Tests are organized by the original paper sections:

- BigSMILES: Lin, Coley, Olsen; ACS Central Science 2019
  DOI: 10.1021/acscentsci.9b00476
- G-BigSMILES: Schneider, Walsh, Olsen, de Pablo; Digital Discovery 2024
  DOI: 10.1039/D3DD00147D

Each test class corresponds to a syntactic feature in the specification.
"""

from molpy.parser.smiles import (
    parse_bigsmiles,
    parse_gbigsmiles,
    bigsmilesir_to_monomer,
    bigsmilesir_to_polymerspec,
)


# =====================================================================
# §2.1  Bonding Descriptors — AA-type ($)
# =====================================================================


class TestAATypeDescriptors:
    """AA-type bonding descriptors: any $n can connect to any other $n.

    Paper §2: "The AA-type descriptor $ allows for non-directional
    bonding between repeat units."
    """

    def test_basic_dollar(self):
        """{[$]CC[$]} — polyethylene, simplest AA-type."""
        ir = parse_bigsmiles("{[$]CC[$]}")
        assert len(ir.stochastic_objects) == 1

    def test_dollar_with_index(self):
        """{[$1]CC[$1]} — indexed AA-type."""
        ir = parse_bigsmiles("{[$1]CC[$1]}")
        assert len(ir.stochastic_objects) == 1

    def test_multiple_dollar_indices(self):
        """{[$1]CC([$2])[$1]} — branching point with two AA indices."""
        ir = parse_bigsmiles("{[$1]CC([$2])[$1]}")
        obj = ir.stochastic_objects[0]
        assert len(obj.repeat_units) >= 1

    def test_aa_monomer_ports(self):
        """AA-type descriptors produce $ ports."""
        ir = parse_bigsmiles("{[$]CC[$]}")
        struct = bigsmilesir_to_monomer(ir)
        ports = [a.get("port") for a in struct.atoms if a.get("port")]
        assert "$" in ports


# =====================================================================
# §2.1  Bonding Descriptors — AB-type (<, >)
# =====================================================================


class TestABTypeDescriptors:
    """AB-type bonding descriptors: < can only bond with >.

    Paper §2: "AB-type descriptors enforce directionality —
    [<] can only connect to [>]."
    """

    def test_basic_ab(self):
        """{[<]CC[>]} — simplest AB-type monomer."""
        ir = parse_bigsmiles("{[<]CC[>]}")
        assert len(ir.stochastic_objects) == 1

    def test_ab_with_index(self):
        """{[<1]CC[>1]} — indexed AB-type."""
        ir = parse_bigsmiles("{[<1]CC[>1]}")
        assert len(ir.stochastic_objects) == 1

    def test_ab_monomer_directional_ports(self):
        """AB-type descriptors create < and > ports."""
        ir = parse_bigsmiles("{[<]CC[>]}")
        struct = bigsmilesir_to_monomer(ir)
        ports = {a.get("port") for a in struct.atoms if a.get("port")}
        assert "<" in ports
        assert ">" in ports


# =====================================================================
# §3  Polymer Architectures — Homopolymer
# =====================================================================


class TestHomopolymer:
    """Homopolymer: single repeat unit.

    Paper Table 1, Figure 2.
    """

    def test_polyethylene(self):
        """{[$]CC[$]} — polyethylene."""
        ir = parse_bigsmiles("{[$]CC[$]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "homopolymer"
        assert len(spec.segments[0].monomers) == 1

    def test_peo_ab_type(self):
        """{[<]OCC[>]} — poly(ethylene oxide) with AB-type."""
        ir = parse_bigsmiles("{[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "homopolymer"


# =====================================================================
# §3  Polymer Architectures — Random Copolymer
# =====================================================================


class TestRandomCopolymer:
    """Random copolymer: multiple repeat units in one stochastic object.

    Paper Table 1: "poly(ethylene-co-1-butene)"
    """

    def test_ethylene_butene(self):
        """{[$]CC[$],[$]CC(CC)[$]} — poly(ethylene-co-1-butene)."""
        ir = parse_bigsmiles("{[$]CC[$],[$]CC(CC)[$]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "random_copolymer"
        assert len(spec.segments[0].monomers) == 2

    def test_three_component_copolymer(self):
        """{[$]CC[$],[$]CC(C)[$],[$]CC(CC)[$]} — terpolymer."""
        ir = parse_bigsmiles("{[$]CC[$],[$]CC(C)[$],[$]CC(CC)[$]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "random_copolymer"
        assert len(spec.segments[0].monomers) == 3


# =====================================================================
# §3  Polymer Architectures — Alternating Copolymer
# =====================================================================


class TestAlternatingCopolymer:
    """Alternating copolymer: AB-type enforces strict alternation.

    Paper Table 1: uses AB-type to separate donor/acceptor monomers.
    """

    def test_ab_alternating(self):
        """{[<]C(=O)c1ccc(cc1)C(=O)[<],[>]OCCO[>]} — PET-like."""
        ir = parse_bigsmiles("{[<]C(=O)c1ccc(cc1)C(=O)[<],[>]OCCO[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert len(spec.segments[0].monomers) == 2

    def test_nylon66(self):
        """{[<]C(=O)CCCCC(=O)[<],[>]NCCCCCCN[>]} — nylon-6,6."""
        ir = parse_bigsmiles("{[<]C(=O)CCCCC(=O)[<],[>]NCCCCCCN[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert len(spec.segments[0].monomers) == 2


# =====================================================================
# §3  Polymer Architectures — Block Copolymer
# =====================================================================


class TestBlockCopolymer:
    """Block copolymer: sequential stochastic objects.

    Paper Table 1: "PEG-b-PPG"
    """

    def test_diblock(self):
        """{[<]OCC[>]}{[<]OC(C)C[>]} — PEG-b-PPG diblock."""
        ir = parse_bigsmiles("{[<]OCC[>]}{[<]OC(C)C[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "block_copolymer"
        assert len(spec.segments) == 2

    def test_triblock(self):
        """{[<]OCC[>]}{[<]OC(C)C[>]}{[<]OCC[>]} — ABA triblock."""
        ir = parse_bigsmiles("{[<]OCC[>]}{[<]OC(C)C[>]}{[<]OCC[>]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert spec.topology == "block_copolymer"
        assert len(spec.segments) == 3


# =====================================================================
# §3  End Groups
# =====================================================================


class TestEndGroups:
    """End group syntax: explicit outside {} or implicit inside with ;.

    Paper §2.3.
    """

    def test_explicit_end_groups(self):
        """C{[$]CC[$]}C — methyl end groups outside stochastic object."""
        ir = parse_bigsmiles("C{[$]CC[$]}C")
        assert len(ir.stochastic_objects) == 1
        # Backbone should contain the end groups
        assert len(ir.backbone.atoms) > 0

    def test_implicit_end_groups_semicolon(self):
        """End group after semicolon (terminal brackets required).

        Paper §2.3: items after ; are end groups, not repeat units.
        """
        ir = parse_bigsmiles("{[][<]CC[>];[<]O[]}")
        obj = ir.stochastic_objects[0]
        assert len(obj.end_groups) >= 1

    def test_ab_with_explicit_start(self):
        """O{[>][<]C(=O)CN[>]} — explicit start group + AB-type."""
        ir = parse_bigsmiles("O{[>][<]C(=O)CN[>]}")
        assert len(ir.backbone.atoms) > 0


# =====================================================================
# §3  Branched Polymers
# =====================================================================


class TestBranchedPolymers:
    """Branched polymers: repeat units with 3+ bonding descriptors.

    Paper §3.2: "Repeat units with three or more bonding descriptors
    serve as branch points."
    """

    def test_three_port_branch_point(self):
        """{[$]CC([$])[$]} — 3-port branch point."""
        ir = parse_bigsmiles("{[$]CC([$])[$]}")
        struct = bigsmilesir_to_monomer(ir)
        ports = [a for a in struct.atoms if a.get("port")]
        assert len(ports) >= 2  # At least 2 ports visible

    def test_ldpe_like(self):
        """{[$]CC[$],[$]CC([$])[$]} — LDPE with linear + branch units."""
        ir = parse_bigsmiles("{[$]CC[$],[$]CC([$])[$]}")
        spec = bigsmilesir_to_polymerspec(ir)
        assert len(spec.segments[0].monomers) == 2


# =====================================================================
# §3  Nested Stochastic Objects (Graft Copolymers)
# =====================================================================


class TestNestedStochasticObjects:
    """Nested stochastic objects for graft and bottlebrush architectures.

    Paper §3.3: "Stochastic objects can appear inside repeat units
    of other stochastic objects."
    """

    def test_graft_copolymer(self):
        """Nested stochastic object: graft architecture."""
        ir = parse_bigsmiles("{[$]CC(c1ccccc1)[$],[$]CC({[$]CC(C)[$]})[$]}")
        assert len(ir.stochastic_objects) >= 1


# =====================================================================
# §3  Ring-Containing Repeat Units
# =====================================================================


class TestRingRepeatUnits:
    """Repeat units containing rings.

    Paper: aromatic and non-aromatic rings in monomers.
    """

    def test_styrene(self):
        """{[$]CC(c1ccccc1)[$]} — polystyrene."""
        ir = parse_bigsmiles("{[$]CC(c1ccccc1)[$]}")
        struct = bigsmilesir_to_monomer(ir)
        # 8 atoms: 2 backbone C + 6 ring C
        assert len(list(struct.atoms)) == 8

    def test_vinyl_pyridine(self):
        """{[$]CC(c1ccncc1)[$]} — poly(vinyl pyridine)."""
        ir = parse_bigsmiles("{[$]CC(c1ccncc1)[$]}")
        struct = bigsmilesir_to_monomer(ir)
        # Should have N in ring
        elements = [a.get("symbol") or a.get("element") for a in struct.atoms]
        assert "N" in elements or "n" in elements


# =====================================================================
# §3  Polyisoprene Isomers (cis/trans/1,2/3,4 addition)
# =====================================================================


class TestPolyisopreneIsomers:
    """Four polyisoprene microstructures.

    Paper Figure 3: 1,4-cis, 1,4-trans, 1,2-addition, 3,4-addition.
    """

    def test_14_cis(self):
        r"""{[<]C/C=C(\C)C[>]} — 1,4-cis polyisoprene (AB-type)."""
        ir = parse_bigsmiles(r"{[<]C/C=C(\C)C[>]}")
        struct = bigsmilesir_to_monomer(ir)
        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1

    def test_14_trans(self):
        """{[<]C/C=C(/C)C[>]} — 1,4-trans polyisoprene (AB-type)."""
        ir = parse_bigsmiles("{[<]C/C=C(/C)C[>]}")
        struct = bigsmilesir_to_monomer(ir)
        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1

    def test_12_addition(self):
        """{[$]CC(C=C)[$]} — 1,2-addition polyisoprene."""
        ir = parse_bigsmiles("{[$]CC(C=C)[$]}")
        struct = bigsmilesir_to_monomer(ir)
        assert len(list(struct.atoms)) == 4

    def test_34_addition(self):
        """{[$]CC(=C)C[$]} — 3,4-addition polyisoprene."""
        ir = parse_bigsmiles("{[$]CC(=C)C[$]}")
        struct = bigsmilesir_to_monomer(ir)
        assert len(list(struct.atoms)) == 4


# =====================================================================
# §3  Tacticity (chirality in repeat units)
# =====================================================================


class TestTacticity:
    """Tacticity via SMILES chirality.

    Paper Table 1:
    - Isotactic:    fixed @ or @@ in single repeat unit
    - Syndiotactic: alternating @@/@ in a dyad repeat unit
    - Atactic:      no chirality annotation
    """

    def test_isotactic_pp(self):
        """{[<]C[C@@H](C)[>]} — isotactic polypropylene."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 1
        assert chiral[0]["chiral"] == "@@"

    def test_syndiotactic_pp_dyad(self):
        """{[<]C[C@@H](C)C[C@H](C)[>]} — syndiotactic PP dyad."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)C[C@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 2
        configs = sorted([c["chiral"] for c in chiral])
        assert configs == ["@", "@@"]

    def test_atactic_pp(self):
        """{[$]CC(C)[$]} — atactic polypropylene (no chirality)."""
        ir = parse_bigsmiles("{[$]CC(C)[$]}")
        struct = bigsmilesir_to_monomer(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 0


# =====================================================================
# §3  Polyelectrolytes (counter-ions via dot notation)
# =====================================================================


class TestPolyelectrolytes:
    """Polyelectrolytes with counter-ions.

    Paper §3.5: counter-ions included via SMILES dot notation.
    """

    def test_sodium_polyacrylate(self):
        """{[$]CC(C(=O)[O-])[$]} — poly(acrylic acid, sodium salt) repeat."""
        ir = parse_bigsmiles("{[$]CC(C(=O)[O-])[$]}")
        struct = bigsmilesir_to_monomer(ir)
        charged = [a for a in struct.atoms if a.get("charge")]
        assert any(a.get("charge") == -1 for a in charged)


# =====================================================================
# G-BigSMILES: Molecular Weight Distributions
# =====================================================================


class TestGBigSmilesDistributions:
    """G-BigSMILES molecular weight distribution syntax.

    Paper §2.2: |distribution_name(params)|
    """

    def test_flory_schulz(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|flory_schulz(0.9)|")
        assert len(ir.molecules) >= 1

    def test_schulz_zimm(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|schulz_zimm(5000, 4500)|")
        assert len(ir.molecules) >= 1

    def test_gauss(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|gauss(10000, 1000)|")
        assert len(ir.molecules) >= 1

    def test_uniform(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|uniform(5000, 15000)|")
        assert len(ir.molecules) >= 1

    def test_log_normal(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|log_normal(10000, 1.2)|")
        assert len(ir.molecules) >= 1

    def test_poisson(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|poisson(50)|")
        assert len(ir.molecules) >= 1


# =====================================================================
# G-BigSMILES: System Size
# =====================================================================


class TestGBigSmilesSystemSize:
    """G-BigSMILES system size annotation.

    Paper §2.3: |total_molecular_weight|
    """

    def test_system_size_integer(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|1000|")
        assert len(ir.molecules) >= 1

    def test_system_size_scientific(self):
        ir = parse_gbigsmiles("{[<]CC[>]}|5e5|")
        assert len(ir.molecules) >= 1

    def test_distribution_and_size(self):
        """Distribution + system size together."""
        ir = parse_gbigsmiles("{[<]CC[>]}|poisson(50)|.|5e5|")
        assert len(ir.molecules) >= 1


# =====================================================================
# G-BigSMILES: Bond Descriptor Weights
# =====================================================================


class TestGBigSmilesDescriptorWeights:
    """G-BigSMILES transition probability weights on descriptors.

    Paper §2.1: [symbol|weight(s)|]
    """

    def test_single_weight(self):
        """[<|3.0|]CC[>|2.0|] — single weight per descriptor."""
        ir = parse_gbigsmiles("{[<|3.0|]CC[>|2.0|]}|poisson(10)|")
        assert len(ir.molecules) >= 1

    def test_weight_list(self):
        """Weight list for transition probabilities."""
        # Two repeat units with weight lists encoding alternation
        ir = parse_gbigsmiles("{[<|0 1|]CC[>|1 0|],[<|1 0|]CC(C)[>|0 1|]}|poisson(10)|")
        assert len(ir.molecules) >= 1


# =====================================================================
# G-BigSMILES: Complex Examples from Paper
# =====================================================================


class TestGBigSmilesComplexExamples:
    """Complex G-BigSMILES strings from the paper."""

    def test_ps_pmma_block(self):
        """PS-PMMA with Schulz-Zimm distribution."""
        ir = parse_gbigsmiles(
            "{[<]CC(c1ccccc1)[>]}|schulz_zimm(5000, 4500)|"
            "{[<]CC(C(=O)OC)[>]}|schulz_zimm(3000, 2800)|"
        )
        assert len(ir.molecules) >= 1

    def test_with_end_groups_and_distribution(self):
        """End groups + distribution."""
        ir = parse_gbigsmiles("C{[>][<]CC(c1ccccc1)[>]}|poisson(100)|")
        assert len(ir.molecules) >= 1
