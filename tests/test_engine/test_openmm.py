"""Tests for OpenMMEngine and OpenMMSimulationConfig."""

import json

import numpy as np
import pytest

from molpy.core import Block, Frame
from molpy.ff.forcefield import ForceField
from molpy.engine import OpenMMEngine, OpenMMSimulationConfig


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_frame():
    """Minimal Frame with three atoms and x/y/z coordinates."""
    frame = Frame()
    atoms = Block(
        {
            "x": np.array([0.0, 1.0, 0.0]),
            "y": np.array([0.0, 0.0, 1.0]),
            "z": np.array([0.0, 0.0, 0.0]),
            # PDB atom names are unique within a residue — two atoms both
            # named "H" make a file readers reject as duplicated.
            "name": np.array(["O", "H1", "H2"], dtype="U4"),
            "element": np.array(["O", "H", "H"], dtype="U2"),
        }
    )
    frame["atoms"] = atoms
    return frame


@pytest.fixture
def empty_forcefield():
    """Minimal ForceField (no types) for XML serialisation tests."""
    return ForceField("test")


@pytest.fixture
def nvt_config():
    return OpenMMSimulationConfig(ensemble="NVT", n_steps=100)


@pytest.fixture
def engine():
    return OpenMMEngine(check_executable=False)


# ---------------------------------------------------------------------------
# OpenMMSimulationConfig
# ---------------------------------------------------------------------------


class TestOpenMMSimulationConfig:
    def test_default_values(self):
        cfg = OpenMMSimulationConfig()
        assert cfg.ensemble == "NVT"
        assert cfg.temperature == 300.0
        assert cfg.timestep_fs == 2.0
        assert cfg.n_steps == 500_000
        assert cfg.nonbonded_method == "PME"
        assert cfg.constraints == "HBonds"
        assert cfg.platform == "CUDA"

    def test_npt_ensemble(self):
        cfg = OpenMMSimulationConfig(ensemble="NPT", pressure=2.0)
        assert cfg.ensemble == "NPT"
        assert cfg.pressure == 2.0

    def test_minimize_ensemble(self):
        cfg = OpenMMSimulationConfig(ensemble="minimize")
        assert cfg.ensemble == "minimize"

    def test_to_dict_returns_dict(self):
        cfg = OpenMMSimulationConfig(n_steps=1000)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["n_steps"] == 1000
        assert d["ensemble"] == "NVT"

    def test_to_dict_roundtrip(self):
        cfg = OpenMMSimulationConfig(temperature=350.0, n_steps=2000)
        restored = OpenMMSimulationConfig.from_dict(cfg.to_dict())
        assert restored.temperature == 350.0
        assert restored.n_steps == 2000

    def test_from_dict(self):
        d = {"ensemble": "NPT", "temperature": 400.0, "n_steps": 5000}
        cfg = OpenMMSimulationConfig.from_dict(d)
        assert cfg.ensemble == "NPT"
        assert cfg.temperature == 400.0

    def test_to_json_creates_file(self, tmp_path):
        cfg = OpenMMSimulationConfig(n_steps=999)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        assert path.exists()

    def test_to_json_roundtrip(self, tmp_path):
        cfg = OpenMMSimulationConfig(temperature=280.0, n_steps=1234)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        restored = OpenMMSimulationConfig.from_json(path)
        assert restored.temperature == 280.0
        assert restored.n_steps == 1234

    def test_json_file_is_valid_json(self, tmp_path):
        cfg = OpenMMSimulationConfig()
        path = tmp_path / "config.json"
        cfg.to_json(path)
        data = json.loads(path.read_text())
        assert "ensemble" in data
        assert "temperature" in data


# ---------------------------------------------------------------------------
# OpenMMEngine initialisation
# ---------------------------------------------------------------------------


class TestOpenMMEngineInit:
    def test_name(self, engine):
        assert engine.name == "OpenMM"

    def test_extension(self, engine):
        assert engine._get_default_extension() == ".py"

    def test_default_executable(self, engine):
        assert engine.executable == "python"

    def test_check_executable_false_does_not_raise(self):
        OpenMMEngine(executable="nonexistent_binary_xyz", check_executable=False)


# ---------------------------------------------------------------------------
# generate_inputs — no OpenMM required
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# write_openmm_system factory
# ---------------------------------------------------------------------------
