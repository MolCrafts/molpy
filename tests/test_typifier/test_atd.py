"""AtdTypifier (ATOMTYPE tables) — hardcoded goldens from am1bcc_reference.

Source: ``molrs-cxxapi/tests/am1bcc_reference.rs`` case ``methane`` (offline
AmberTools dump). Tests never spawn external tools.

# subset reason: methane only — full 37-molecule matrix is the cxxapi
# ``am1bcc_reference`` fixture + ``am1bcc_bridge``; this file pins the Python
# AtdTypifier binding on one neutral, fully-covered molecule so every table
# returns a complete type string list (no missing-type escape hatch).
"""

from __future__ import annotations

import numpy as np

import molrs

# ATOMTYPE table codes for methane
_METHANE_GFF = ["c3", "hc", "hc", "hc", "hc"]
_METHANE_GAS = ["c3", "h", "h", "h", "h"]
_METHANE_AMBER = ["CT", "HC", "HC", "HC", "HC"]
_METHANE_SYBYL = ["C.3", "H", "H", "H", "H"]
_METHANE_BCC = ["11", "91", "91", "91", "91"]
# Gasteiger / gas charges (float64), same fixture
_METHANE_GAS_CHARGES = np.array(
    [-0.077576, 0.019394, 0.019394, 0.019394, 0.019394], dtype=np.float64
)


def _methane() -> molrs.Atomistic:
    mol = molrs.io.SmilesIR("C").to_atomistic()
    return molrs.perceive.Perceive().find_hydrogens(mol)


def _types(parameter_set: str) -> list[str]:
    typed = molrs.ff.typifier.AtdTypifier(parameter_set=parameter_set).typify(
        _methane()
    )
    return list(typed.to_frame()["atoms"]["type"])


def test_methane_gaff_types():
    assert _types("gaff") == _METHANE_GFF


def test_methane_gaff2_types():
    assert _types("gaff2") == _METHANE_GFF


def test_methane_gas_types():
    assert _types("gas") == _METHANE_GAS


def test_methane_amber_types():
    assert _types("amber") == _METHANE_AMBER


def test_methane_sybyl_types():
    assert _types("sybyl") == _METHANE_SYBYL


def test_methane_bcc_types():
    assert _types("bcc") == _METHANE_BCC


def test_methane_gasteiger_charges_match_reference_float64():
    """Python path returns the same gas charges as the reference fixture.

    Fixture decimals are 6 places; require float64 dtype and conservation to 1e-12.
    """
    mol = _methane()
    typed = molrs.ff.typifier.AtdTypifier(parameter_set="gas").typify(mol)
    got = np.asarray(molrs.ff.charge.GasteigerModel().assign(typed), dtype=np.float64)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, _METHANE_GAS_CHARGES, rtol=0.0, atol=1e-6)
    assert abs(float(got.sum())) < 1e-12
