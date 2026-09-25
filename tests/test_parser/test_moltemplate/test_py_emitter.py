"""Unit tests for :mod:`molpy.parser.moltemplate.py_emitter` (``PythonScriptEmitter``).

The emitted script is executed in-process (``runpy``) — no subprocess.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from molpy.parser.moltemplate import (
    MolTemplateBuilder,
    PythonScriptEmitter,
    parse_file,
)


def _coordinates(system) -> list[tuple[float, float, float]]:
    return [(float(a["x"]), float(a["y"]), float(a["z"])) for a in system.atoms]


def _run_emitted(lt_path: Path, dest: Path):
    script = PythonScriptEmitter(base_dir=lt_path.parent).emit(
        parse_file(lt_path), dest
    )
    namespace = runpy.run_path(str(script))
    system, _ff = namespace["build_system"]()
    return system


class TestPythonScriptEmitterArrayReplication:
    """The generated script replicates ``[N].op(...)`` like the builder does."""

    @pytest.mark.parametrize(
        "fixture", ["array_move.lt", "array_rot.lt", "array_scale.lt"]
    )
    def test_copy_k_carries_the_transform_k_times(
        self, fixture, tmp_path, array_transforms_dir, array_transform_goldens
    ):
        system = _run_emitted(array_transforms_dir / fixture, tmp_path / "script.py")

        got = _coordinates(system)
        want = array_transform_goldens[fixture]
        assert len(got) == len(want)
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            assert g == pytest.approx(w, abs=1e-8), f"copy {k}"

    def test_random_instances_sit_on_the_grid(self, tmp_path, array_transforms_dir):
        system = _run_emitted(
            array_transforms_dir / "random_grid.lt", tmp_path / "script.py"
        )

        anchors = sorted(float(a["x"]) for a in system.atoms if float(a["y"]) == 0.0)
        assert anchors == pytest.approx([0.0, 5.0, 10.0], abs=1e-12)

    def test_seeded_random_matches_the_builder(self, tmp_path, array_transforms_dir):
        """Same seed, same document: the script draws what the builder draws.

        The seeded draw has no hand-derivable golden, so the builder's result is
        the reference here.
        """
        lt_path = array_transforms_dir / "random_grid.lt"
        system = _run_emitted(lt_path, tmp_path / "script.py")
        reference, _ = MolTemplateBuilder(
            parse_file(lt_path), base_dir=lt_path.parent
        ).build_system()

        assert _coordinates(system) == pytest.approx(_coordinates(reference), abs=1e-12)


class TestPythonScriptEmitterEmit:
    """``emit`` resolves imports against the ``base_dir`` held by the emitter."""

    def test_imports_resolve_relative_to_constructor_base_dir(
        self, tmp_path, monkeypatch, TEST_DATA_DIR
    ):
        # cwd is elsewhere, so only the constructor's base_dir can find the
        # imported ``polymer.lt`` -> ``monomer.lt`` -> ``forcefield.lt`` chain.
        monkeypatch.chdir(tmp_path)
        fixture_dir = TEST_DATA_DIR / "moltemplate" / "2bead_polymer"

        script = PythonScriptEmitter(base_dir=fixture_dir).emit(
            parse_file(fixture_dir / "system.lt"), tmp_path / "script.py"
        )
        system, _ff = runpy.run_path(str(script))["build_system"]()

        # 27 polymers (3x3x3 grid) x 7 monomers x 2 beads (ca, r).
        assert len(list(system.atoms)) == 27 * 7 * 2
