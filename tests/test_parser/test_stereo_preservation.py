"""Tests for stereochemistry preservation in SMILES/BigSMILES/CGSmiles.

Stereochemistry in MolPy follows the original paper definitions:

- **SMILES**: Standard @ / @@ chirality and / \\ bond stereo
  (OpenSMILES spec)

- **BigSMILES** (Lin, Coley, Olsen; ACS Central Sci 2019):
  Tacticity is encoded within repeat unit SMILES, not as keywords.
  - Isotactic PP:    {[<]C[C@@H](C)[>]}          (fixed chirality)
  - Syndiotactic PP: {[<]C[C@@H](C)C[C@H](C)[>]} (dyad, alternating)
  - Atactic PP:      {[$]CC(C)[$]}                (no chirality)
  - AB-type descriptors (</>)  enforce head-to-tail orientation
  - AA-type descriptors ($) allow orientation flipping

- **G-BigSMILES** (Digital Discovery, RSC, 2024):
  Transition probability matrices in descriptors control alternation:
    [<|0 0 0 1|]C[C@H](C)[>|0 0 1 0|],
    [<|0 1 0 0|]C[C@@H](C)[>|1 0 0 0|]

- **CGSmiles** (Grunewald et al.; JCIM 2025):
  Explicit R/S annotation via key-value pairs: [C;x=S], [C;x=R]

These tests verify that chirality and bond stereo information survive
the full parsing pipeline: string → grammar → IR → converter → Atomistic.
"""

from molpy.parser.smiles import (
    parse_smiles,
    parse_bigsmiles,
    smilesir_to_atomistic,
    bigsmilesir_to_monomer,
    bigsmilesir_to_polymerspec,
)


# =====================================================================
# SMILES-level: basic stereo preservation through converter
# =====================================================================


class TestSmilesChiralPreservation:
    """Verify @ / @@ survive parse_smiles → smilesir_to_atomistic."""

    def test_at_preserved(self):
        ir = parse_smiles("[C@H](C)(N)O")
        struct = smilesir_to_atomistic(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 1
        assert chiral[0]["chiral"] == "@"

    def test_atat_preserved(self):
        ir = parse_smiles("[C@@H](C)(N)O")
        struct = smilesir_to_atomistic(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 1
        assert chiral[0]["chiral"] == "@@"

    def test_multiple_chiral_centers(self):
        ir = parse_smiles("[C@H](C)[C@@H](C)C")
        struct = smilesir_to_atomistic(ir)
        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 2
        assert chiral[0]["chiral"] == "@"
        assert chiral[1]["chiral"] == "@@"


class TestSmilesBondStereoPreservation:
    """Verify / and \\ survive parse_smiles → smilesir_to_atomistic."""

    def test_forward_slash(self):
        ir = parse_smiles("C/C=C/C")
        struct = smilesir_to_atomistic(ir)
        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1
        assert "/" in [b["stereo"] for b in stereo]

    def test_backslash(self):
        ir = parse_smiles(r"C\C=C\C")
        struct = smilesir_to_atomistic(ir)
        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1
        assert "\\" in [b["stereo"] for b in stereo]

    def test_mixed_ez(self):
        """E-configuration: C/C=C\\C has both / and \\."""
        ir = parse_smiles(r"C/C=C\C")
        struct = smilesir_to_atomistic(ir)
        stereo = [b for b in struct.bonds if b.get("stereo")]
        symbols = {b["stereo"] for b in stereo}
        assert "/" in symbols
        assert "\\" in symbols

    def test_combined_chiral_and_bond_stereo(self):
        """Both features co-exist on one molecule."""
        ir = parse_smiles("[C@H](/C=C/C)(C)O")
        struct = smilesir_to_atomistic(ir)
        assert any(a.get("chiral") for a in struct.atoms)
        assert any(b.get("stereo") for b in struct.bonds)


# =====================================================================
# BigSMILES-level: tacticity per the paper (Lin et al., 2019)
# =====================================================================


class TestBigSmilesIsotacticPP:
    """Isotactic polypropylene — single repeat unit with fixed chirality.

    BigSMILES: {[<]C[C@@H](C)[>]}
    Paper ref: Table 1, "isotactic polypropylene"
    The AB-type descriptors (</>)  enforce head-to-tail.
    """

    def test_isotactic_monomer_chirality(self):
        """Repeat unit C[C@@H](C) should carry chiral == '@@'."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)

        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 1, "Isotactic repeat unit has one chiral center"
        assert chiral[0]["chiral"] == "@@"

    def test_isotactic_monomer_ports(self):
        """AB-type descriptors create < and > ports."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)

        ports = [a.get("port") for a in struct.atoms if a.get("port")]
        assert "<" in ports
        assert ">" in ports

    def test_isotactic_enantiomer(self):
        """The other enantiomer uses @."""
        ir = parse_bigsmiles("{[<]C[C@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)

        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 1
        assert chiral[0]["chiral"] == "@"


class TestBigSmilesSyndiotacticPP:
    """Syndiotactic polypropylene — dyad with alternating @@ / @.

    BigSMILES: {[<]C[C@@H](C)C[C@H](C)[>]}
    Paper ref: Table 1, "syndiotactic polypropylene"
    A single repeat unit encodes the dyad; AB-descriptors enforce orientation.
    """

    def test_syndiotactic_dyad_chirality(self):
        """Dyad repeat unit has two chiral centers with opposite config."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)C[C@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)

        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 2, "Syndiotactic dyad has two chiral centers"
        configs = sorted([c["chiral"] for c in chiral])
        assert configs == ["@", "@@"], "One @ and one @@ in the dyad"

    def test_syndiotactic_dyad_ports(self):
        """Dyad still has < and > ports."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)C[C@H](C)[>]}")
        struct = bigsmilesir_to_monomer(ir)

        ports = [a.get("port") for a in struct.atoms if a.get("port")]
        assert "<" in ports
        assert ">" in ports


class TestBigSmilesAtacticPP:
    """Atactic polypropylene — no chirality annotation.

    BigSMILES: {[$]CC(C)[$]}
    Paper ref: Table 1, "polypropylene"
    AA-type descriptors ($) allow orientation flipping — atactic by default.
    """

    def test_atactic_no_chirality(self):
        """Atactic repeat unit has NO chiral centers."""
        ir = parse_bigsmiles("{[$]CC(C)[$]}")
        struct = bigsmilesir_to_monomer(ir)

        chiral = [a for a in struct.atoms if a.get("chiral")]
        assert len(chiral) == 0, "Atactic PP has no chiral annotation"

    def test_atactic_aa_descriptor(self):
        """AA-type uses $ ports (allow flipping)."""
        ir = parse_bigsmiles("{[$]CC(C)[$]}")
        struct = bigsmilesir_to_monomer(ir)

        ports = [a.get("port") for a in struct.atoms if a.get("port")]
        assert "$" in ports


class TestBigSmilesPolymerSpec:
    """PolymerSpec correctly reflects tacticity from BigSMILES."""

    def test_isotactic_homopolymer(self):
        ir = parse_bigsmiles("{[<]C[C@@H](C)[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert spec.topology == "homopolymer"
        assert len(spec.segments) == 1
        assert len(spec.segments[0].monomers) == 1

        monomer = spec.segments[0].monomers[0]
        chiral = [a for a in monomer.atoms if a.get("chiral")]
        assert len(chiral) == 1
        assert chiral[0]["chiral"] == "@@"

    def test_syndiotactic_homopolymer(self):
        ir = parse_bigsmiles("{[<]C[C@@H](C)C[C@H](C)[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert spec.topology == "homopolymer"
        monomer = spec.segments[0].monomers[0]
        chiral = [a for a in monomer.atoms if a.get("chiral")]
        assert len(chiral) == 2


# =====================================================================
# BigSMILES-level: cis/trans polydienes (bond stereo in repeat units)
# =====================================================================


class TestBigSmilesCisTransPolydienes:
    """Bond stereo (E/Z) in BigSMILES repeat units.

    cis-1,4-polybutadiene:  {[<]C/C=C\\C[>]}
    trans-1,4-polybutadiene: {[<]C/C=C/C[>]}
    """

    def test_cis_polybutadiene_bond_stereo(self):
        r"""cis: C/C=C\C — both / and \\ present."""
        ir = parse_bigsmiles(r"{[<]C/C=C\C[>]}")
        struct = bigsmilesir_to_monomer(ir)

        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1, "cis-polydiene has bond stereo"

    def test_trans_polybutadiene_bond_stereo(self):
        """trans: C/C=C/C — both / present."""
        ir = parse_bigsmiles("{[<]C/C=C/C[>]}")
        struct = bigsmilesir_to_monomer(ir)

        stereo = [b for b in struct.bonds if b.get("stereo")]
        assert len(stereo) >= 1, "trans-polydiene has bond stereo"


# =====================================================================
# BigSMILES-level: AB vs AA descriptors
# =====================================================================


class TestBigSmilesDescriptorTypes:
    """AB-type (</>)  vs AA-type ($) descriptor semantics.

    AB-type: <n matches >n only → head-to-tail enforced
    AA-type: $n matches $n → orientation flipping allowed
    These impact whether stereo is preserved or randomized during assembly.
    """

    def test_ab_type_creates_directional_ports(self):
        ir = parse_bigsmiles("{[<]CC[>]}")
        struct = bigsmilesir_to_monomer(ir)
        ports = {a.get("port") for a in struct.atoms if a.get("port")}
        assert "<" in ports and ">" in ports

    def test_aa_type_creates_symmetric_ports(self):
        ir = parse_bigsmiles("{[$]CC[$]}")
        struct = bigsmilesir_to_monomer(ir)
        ports = [a.get("port") for a in struct.atoms if a.get("port")]
        assert all(p == "$" for p in ports)
        assert len(ports) == 2


# =====================================================================
# BigSMILES-level: chirality in copolymers
# =====================================================================


class TestBigSmilesCopolymerStereo:
    """Stereo in copolymer repeat units."""

    def test_stereo_in_multiple_repeat_units(self):
        """Two monomers with different chirality in a random copolymer."""
        ir = parse_bigsmiles("{[<]C[C@@H](C)[>],[<]C[C@H](C)[>]}")
        spec = bigsmilesir_to_polymerspec(ir)

        assert spec.topology == "random_copolymer"
        assert len(spec.segments[0].monomers) == 2

        mon0 = spec.segments[0].monomers[0]
        mon1 = spec.segments[0].monomers[1]

        chiral0 = [a for a in mon0.atoms if a.get("chiral")]
        chiral1 = [a for a in mon1.atoms if a.get("chiral")]

        assert len(chiral0) == 1 and chiral0[0]["chiral"] == "@@"
        assert len(chiral1) == 1 and chiral1[0]["chiral"] == "@"
