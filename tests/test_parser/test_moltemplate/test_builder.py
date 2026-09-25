"""Unit tests for :mod:`molpy.parser.moltemplate.builder` (``MolTemplateBuilder``)."""

from __future__ import annotations

from pathlib import Path

import pytest

import molpy as mp
from molpy.parser.moltemplate import MolTemplateBuilder, parse_file


def _coordinates(system) -> list[tuple[float, float, float]]:
    return [(float(a["x"]), float(a["y"]), float(a["z"])) for a in system.atoms]


def _builder(path: Path) -> MolTemplateBuilder:
    return MolTemplateBuilder(parse_file(path), base_dir=path.parent)


class TestMolTemplateBuilderArrayReplication:
    """``[N].op(...)``: copy k carries ``op`` applied k times, copy 0 untransformed."""

    @pytest.mark.parametrize(
        "fixture", ["array_move.lt", "array_rot.lt", "array_scale.lt"]
    )
    def test_copy_k_carries_the_transform_k_times(
        self, fixture, array_transforms_dir, array_transform_goldens
    ):
        system, _ = _builder(array_transforms_dir / fixture).build_system()

        got = _coordinates(system)
        want = array_transform_goldens[fixture]
        assert len(got) == len(want)
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            assert g == pytest.approx(w, abs=1e-8), f"copy {k}"

    def test_random_instances_sit_on_the_grid(self, array_transforms_dir):
        system, _ = _builder(array_transforms_dir / "random_grid.lt").build_system()

        anchors = sorted(float(a["x"]) for a in system.atoms if float(a["y"]) == 0.0)
        assert anchors == pytest.approx([0.0, 5.0, 10.0], abs=1e-12)


class TestMolTemplateBuilderBuildSystem:
    def test_build_system_populates_a_given_forcefield(self, TEST_DATA_DIR):
        ff = mp.ForceField(name="given", units="real")

        _, got = _builder(
            TEST_DATA_DIR / "moltemplate" / "bonded_trimer.lt"
        ).build_system(ff=ff)

        assert got is ff

    @pytest.mark.parametrize(("auto_topology", "n_angles"), [(True, 1), (False, 0)])
    def test_auto_topology_controls_derived_angles(
        self, TEST_DATA_DIR, auto_topology, n_angles
    ):
        system, _ = _builder(
            TEST_DATA_DIR / "moltemplate" / "bonded_trimer.lt"
        ).build_system(auto_topology=auto_topology)

        assert len(list(system.angles)) == n_angles

    def test_build_forcefield_then_build_system_share_one_forcefield(
        self, TEST_DATA_DIR
    ):
        builder = _builder(TEST_DATA_DIR / "moltemplate" / "bonded_trimer.lt")

        ff = builder.build_forcefield()
        _, system_ff = builder.build_system()

        assert system_ff is ff
