"""Tests for the multi-engine emitter registry and outputs."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from molpy.io.emit import EMITTERS, emit, emit_all
from molpy.io.forcefield.moltemplate import read_moltemplate_system

FIXTURES = (
    Path(__file__).parent.parent.parent
    / "test_parser"
    / "test_moltemplate"
    / "fixtures"
)


@pytest.fixture
def tip3p_system():
    return read_moltemplate_system(FIXTURES / "tip3p.lt")


class TestRegistry:
    def test_all_engines_registered(self):
        assert set(EMITTERS) >= {"lammps", "openmm", "gromacs", "xml"}


class TestLammps:
    def test_emit_produces_four_files(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        paths = emit("lammps", atomistic, ff, tmp_path, prefix="w")
        names = {p.name for p in paths}
        assert names == {"w.data", "w.in.settings", "w.in.init", "w.in"}
        assert (tmp_path / "w.in").read_text().startswith("#")

    def test_in_init_has_units(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        emit("lammps", atomistic, ff, tmp_path, prefix="w")
        assert "units real" in (tmp_path / "w.in.init").read_text()


class TestOpenMM:
    def test_emit_produces_three_files(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        paths = emit("openmm", atomistic, ff, tmp_path, prefix="w")
        names = {p.name for p in paths}
        assert names == {"w.xml", "w.pdb", "w.py"}

    def test_py_script_is_valid_python(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        emit("openmm", atomistic, ff, tmp_path, prefix="w")
        ast.parse((tmp_path / "w.py").read_text())


class TestGromacs:
    def test_emit_produces_four_files(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        paths = emit("gromacs", atomistic, ff, tmp_path, prefix="w")
        names = {p.name for p in paths}
        assert names == {"w.gro", "w.top", "em.mdp", "nvt.mdp"}

    def test_mdp_has_integrator(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        emit("gromacs", atomistic, ff, tmp_path, prefix="w")
        assert "integrator" in (tmp_path / "em.mdp").read_text()


class TestXMLEmitter:
    def test_emit_produces_two_files(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        paths = emit("xml", atomistic, ff, tmp_path, prefix="w")
        names = {p.name for p in paths}
        assert names == {"w.xml", "w.pdb"}


class TestEmitAll:
    def test_runs_every_engine(self, tmp_path, tip3p_system):
        atomistic, ff = tip3p_system
        results = emit_all(atomistic, ff, tmp_path, prefix="w")
        assert set(results) >= {"lammps", "openmm", "gromacs", "xml"}
        for engine, paths in results.items():
            for p in paths:
                assert p.exists(), f"{engine} did not write {p}"
