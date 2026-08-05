"""Unit tests for :mod:`molpy.builder.ambertools` — no AmberTools binary."""

from __future__ import annotations

import pytest

import molpy as mp
from molpy.builder.ambertools import AmberResult, AmberTools


class TestAmberResult:
    def test_forcefield_field_only(self):
        forcefield = object()
        result = AmberResult(frame=object(), forcefield=forcefield)
        assert result.forcefield is forcefield
        assert not hasattr(result, "ff")


class TestAmberTools:
    def test_default_uses_system_environment(self, tmp_path):
        amber = AmberTools(work_dir=tmp_path)
        assert amber.env is None
        assert amber.env_manager is None

    def test_env_requires_manager(self, tmp_path):
        with pytest.raises(ValueError, match="incomplete"):
            AmberTools(env="AmberTools25", work_dir=tmp_path)

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
        assert amber.env == "AmberTools25"
        assert amber.env_manager == "conda"
        assert amber._polymer_builders == {}

    def test_env_manager_without_env_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="incomplete"):
            AmberTools(env_manager="conda", work_dir=tmp_path)

    def test_amber_atom_names_are_added_to_a_copy(self, tmp_path):
        struct = mp.Atomistic()
        struct.def_atom(element="C")
        named = AmberTools(work_dir=tmp_path)._named_copy(struct)
        assert struct.atoms[0].get("name") is None
        assert named.atoms[0].get("name") == "C1"
