"""Tests for ``molpy moltemplate convert <src>.lt <dst>.py``.

The emitted script must be syntactically valid Python and produce the
same atom / bond / angle / dihedral counts as the native builder when
executed.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from molpy.io.forcefield.moltemplate import read_moltemplate_system
from molpy.parser.moltemplate import emit_python, parse_file

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def emit_and_load(tmp_path):
    def _inner(src: Path) -> dict:
        dst = tmp_path / (src.stem + ".py")
        emit_python(parse_file(src), dst, base_dir=src.parent)
        # ``runpy.run_path`` executes the module as ``__main__`` substitute; we
        # bypass the ``if __name__ == "__main__"`` guard so build_system runs.
        return runpy.run_path(str(dst), run_name="_emitted")

    return _inner


def _counts(atomistic):
    return (
        len(list(atomistic.atoms)),
        len(list(atomistic.bonds)),
        len(list(atomistic.angles)),
        len(list(atomistic.dihedrals)),
    )


def test_tip3p_roundtrip(emit_and_load):
    src = FIXTURES / "tip3p.lt"
    mod = emit_and_load(src)
    system, ff = mod["build_system"]()
    expected_system, _ = read_moltemplate_system(src)
    assert _counts(system) == _counts(expected_system)
    assert len(list(system.impropers)) == 0


def test_butane_roundtrip(emit_and_load):
    src = FIXTURES / "butane" / "system.lt"
    mod = emit_and_load(src)
    system, ff = mod["build_system"]()
    expected_system, _ = read_moltemplate_system(src)
    # Atom / bond counts must match exactly.
    assert len(list(system.atoms)) == len(list(expected_system.atoms))
    assert len(list(system.bonds)) == len(list(expected_system.bonds))


def test_cli_convert_to_py(tmp_path):
    src = FIXTURES / "tip3p.lt"
    dst = tmp_path / "tip3p.py"
    result = subprocess.run(
        [sys.executable, "-m", "molpy", "moltemplate", "convert", str(src), str(dst)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert dst.exists()
    assert "build_system" in dst.read_text()


def test_emit_python_helper_returns_resolved_path(tmp_path):
    src = FIXTURES / "tip3p.lt"
    dst = tmp_path / "nested" / "tip3p.py"
    out = emit_python(parse_file(src), dst, base_dir=src.parent)
    assert out.exists()
    assert out.is_absolute()
