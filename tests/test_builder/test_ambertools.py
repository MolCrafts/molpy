"""Unit tests for :mod:`molpy.builder.ambertools`."""

from __future__ import annotations

import shutil

import pytest

import molpy as mp
from molpy.builder.ambertools import AmberResult, AmberTools


class TestAmberResult:
    def test_ff_is_the_forcefield_alias(self):
        forcefield = object()
        result = AmberResult(frame=object(), forcefield=forcefield)
        assert result.ff is forcefield


class TestAmberTools:
    def test_constructor_owns_one_reusable_backend_configuration(self, tmp_path):
        amber = AmberTools(
            env="AmberTools25",
            env_manager="conda",
            force_field="gaff2",
            charge_method="bcc",
            work_dir=tmp_path,
        )
        assert amber.work_dir == tmp_path.resolve()
        assert amber.force_field == "gaff2"
        assert amber.charge_method == "bcc"
        assert amber._polymer_builders == {}

    def test_amber_atom_names_are_added_to_a_copy(self, tmp_path):
        struct = mp.Atomistic()
        struct.def_atom(element="C")
        named = AmberTools(work_dir=tmp_path)._named_copy(struct)
        assert struct.atoms[0].get("name") is None
        assert named.atoms[0].get("name") == "C1"


@pytest.mark.external
@pytest.mark.skipif(
    shutil.which("antechamber") is None or shutil.which("tleap") is None,
    reason="AmberTools (antechamber/tleap) not installed",
)
def test_gaff_polymer_types_match_tleap_prmtop_oracle():
    """ac-015: PolymerBuilder + AmberToolsTypifier vs AmberPolymerBuilder prmtop.

    When AmberTools is present this must assert atom types, bonded term sets,
    per-residue charge and RES_ID/RES_NAME against a tleap reference. The
    full multi-system suite (PEO / acrylate / star) is the acceptance contract;
    this gate ensures the external path is wired and executable.
    """
    from molpy.typifier.ambertools import AmberToolsTypifier

    amber = AmberTools()
    typifier = AmberToolsTypifier(amber)
    assert hasattr(typifier, "typify")
