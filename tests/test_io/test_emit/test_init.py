"""``molpy.io.emit`` — one emitter registry, reached through ``emitters``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import molpy.io.emit as emit
from molpy.io.emit import Emitter, EmitterRegistry


class _RecordingEmitter(Emitter):
    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def emit(
        self, atomistic, ff, out_dir: Path, *, prefix: str = "system", **opts: Any
    ):
        self.calls.append((out_dir, prefix))
        return [out_dir / f"{prefix}.rec"]


class TestModuleSurface:
    def test_there_is_no_module_level_emitter_dict(self):
        assert not hasattr(emit, "EMITTERS")

    def test_there_is_no_free_register_function(self):
        assert not hasattr(emit, "register")

    def test_built_in_emitters_live_on_the_one_registry(self):
        assert isinstance(emit.emitters, EmitterRegistry)
        assert emit.emitters.names() == ["gromacs", "lammps", "openmm", "xml"]


class TestEmitterRegistry:
    def test_register_then_names_round_trips(self):
        registry = EmitterRegistry()
        registry.register("rec", _RecordingEmitter())
        assert registry.names() == ["rec"]

    def test_register_then_emit_dispatches_to_that_emitter(
        self, tmp_path, water, tip3p
    ):
        registry = EmitterRegistry()
        recorder = _RecordingEmitter()
        registry.register("rec", recorder)

        paths = registry.emit("rec", water, tip3p, tmp_path / "out", prefix="w")

        assert paths == [tmp_path / "out" / "w.rec"]
        assert recorder.calls == [(tmp_path / "out", "w")]

    def test_registering_on_one_registry_does_not_touch_the_built_ins(self):
        EmitterRegistry().register("rec", _RecordingEmitter())
        assert "rec" not in emit.emitters.names()
