"""Bonded-term (bond/angle/dihedral) typing must assign the CORRECT parameters.

Regression guard for a class of bugs where the OPLS/CL&P force fields store
bond/angle/dihedral types with wildcard atomtype objects (the class pair lives
only in the type *name*, e.g. ``"OW-HW"``). A naive atomtype-object matcher then
returns the first wildcard type for every term, collapsing every bond to TIP3P
water (k=600, r0=0.9572). These tests assert real, distinct parameters — not just
that *a* type was assigned (which the older tests checked).
"""

import warnings

import pytest

import molpy as mp
from molpy.typifier import ClpTypifier, OplsTypifier
from molpy.io.forcefield.xml import read_oplsaa_forcefield

# TIP3P water O-H bond — the value every term wrongly collapsed to.
WATER_BOND_K, WATER_BOND_R0 = 600.0, 0.9572


def _embed(smiles: str) -> mp.Atomistic:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mol = mp.parser.parse_molecule(smiles)
        mol = mp.adapter.generate_3d(mol, add_hydrogens=True, optimize=True)
        return mol.get_topo(gen_angle=True, gen_dihe=True)


def _bond_set(struct):
    """{(frozenset{type_i,type_j}): (k, r0)} for every typed bond."""
    out = {}
    for b in struct.bonds:
        key = frozenset((b.itom.get("type"), b.jtom.get("type")))
        out[key] = (round(float(b.get("k")), 3), round(float(b.get("r0")), 4))
    return out


# --------------------------------------------------------------------------
# OPLS-AA solvent bonds get real OPLS params, never collapsed to water
# --------------------------------------------------------------------------
def test_emc_bonds_not_collapsed_to_water():
    ff = read_oplsaa_forcefield("oplsaa.xml")
    emc = OplsTypifier(ff, strict_typing=True).typify(_embed("CCOC(=O)OC"))
    bonds = _bond_set(emc)
    # No EMC bond is the water O-H bond.
    for k, r0 in bonds.values():
        assert (k, r0) != (WATER_BOND_K, WATER_BOND_R0), (
            f"bond collapsed to TIP3P water params {bonds}"
        )


def test_emc_carbonyl_and_backbone_bond_values():
    ff = read_oplsaa_forcefield("oplsaa.xml")
    emc = OplsTypifier(ff, strict_typing=True).typify(_embed("CCOC(=O)OC"))
    bonds = _bond_set(emc)
    # Carbonyl C_2=O_2 (opls_465-opls_466): k=570, r0=1.229.
    assert bonds[frozenset(("opls_465", "opls_466"))] == (570.0, 1.229)
    # Ester C_2-OS (opls_465-opls_467): k=214, r0=1.327.
    assert bonds[frozenset(("opls_465", "opls_467"))] == (214.0, 1.327)
    # Ethyl CT-CH2 (opls_135-opls_490): k=268, r0=1.529.
    assert bonds[frozenset(("opls_135", "opls_490"))] == (268.0, 1.529)


# --------------------------------------------------------------------------
# CL&P anion bonds get CL&P params (overlay), never water or OPLS fallbacks
# --------------------------------------------------------------------------
def test_fsi_bonds_are_clp_not_water():
    fsi = ClpTypifier().typify(_embed("[N-](S(=O)(=O)F)S(=O)(=O)F"))
    bonds = _bond_set(fsi)
    for k, r0 in bonds.values():
        assert (k, r0) != (WATER_BOND_K, WATER_BOND_R0)
    # NB-SB (NBT-SBT): clp.xml k=313700 kJ/mol/nm^2 -> 374.88 kcal/mol/A^2; r0=1.57 A.
    assert bonds[frozenset(("NBT", "SBT"))] == (374.88, 1.57)
    # SB-OB (SBT-OBT): clp.xml k=533100 -> 637.07; r0=1.437 A.
    assert bonds[frozenset(("SBT", "OBT"))] == (637.07, 1.437)
    # FB-SB (FSI-SBT): clp.xml k=187900 -> 224.546; r0=1.575 A.
    assert bonds[frozenset(("FSI", "SBT"))] == (224.546, 1.575)


# --------------------------------------------------------------------------
# Angles likewise are not collapsed to the first wildcard angle (water HOH)
# --------------------------------------------------------------------------
def test_emc_angles_distinct():
    ff = read_oplsaa_forcefield("oplsaa.xml")
    emc = OplsTypifier(ff, strict_typing=True).typify(_embed("CCOC(=O)OC"))
    from molpy.core.atomistic import Angle

    thetas = {
        round(float(a.get("theta0") or a.get("angle") or 0.0), 4)
        for a in emc.links.bucket(Angle)
    }
    # A real molecule with sp3 + sp2 + ether centres has more than one unique
    # equilibrium angle; a collapse-to-water bug yields exactly one.
    assert len(thetas) > 1, f"all angles collapsed to a single type: {thetas}"
