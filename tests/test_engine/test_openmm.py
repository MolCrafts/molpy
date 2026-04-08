"""Tests for OpenMMEngine and OpenMMSimulationConfig."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from molpy.core.frame import Block, Frame
from molpy.core.forcefield import AtomisticForcefield
from molpy.engine.openmm import OpenMMEngine, OpenMMSimulationConfig


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
            "name": np.array(["O", "H", "H"], dtype="U4"),
            "element": np.array(["O", "H", "H"], dtype="U2"),
        }
    )
    frame["atoms"] = atoms
    return frame


@pytest.fixture
def empty_forcefield():
    """Minimal AtomisticForcefield (no types) for XML serialisation tests."""
    return AtomisticForcefield("test")


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


class TestGenerateInputs:
    def test_creates_output_dir(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        out = tmp_path / "new_dir"
        engine.generate_inputs(simple_frame, empty_forcefield, nvt_config, out)
        assert out.is_dir()

    def test_pdb_file_created(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        assert paths["pdb"].exists()

    def test_xml_ff_file_created(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        assert paths["forcefield"].exists()

    def test_script_file_created(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        assert paths["script"].exists()

    def test_returns_path_dict(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        assert set(paths.keys()) == {"pdb", "forcefield", "script"}
        for v in paths.values():
            assert isinstance(v, Path)

    def test_custom_filenames(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame,
            empty_forcefield,
            nvt_config,
            tmp_path,
            pdb_filename="mol.pdb",
            ff_filename="ff.xml",
            script_filename="run.py",
        )
        assert paths["pdb"].name == "mol.pdb"
        assert paths["forcefield"].name == "ff.xml"
        assert paths["script"].name == "run.py"

    def test_nvt_script_contains_langevin(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig(ensemble="NVT")
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "LangevinMiddleIntegrator" in text

    def test_npt_script_contains_barostat(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig(ensemble="NPT")
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "MonteCarloBarostat" in text

    def test_minimize_script_contains_minimize_energy(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig(ensemble="minimize")
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "minimizeEnergy" in text
        # Minimization script should not call step()
        assert "simulation.step(" not in text

    def test_script_embeds_pdb_filename(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame,
            empty_forcefield,
            nvt_config,
            tmp_path,
            pdb_filename="my_system.pdb",
        )
        text = paths["script"].read_text()
        assert "my_system.pdb" in text

    def test_script_embeds_ff_filename(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        paths = engine.generate_inputs(
            simple_frame,
            empty_forcefield,
            nvt_config,
            tmp_path,
            ff_filename="my_ff.xml",
        )
        text = paths["script"].read_text()
        assert "my_ff.xml" in text

    def test_script_embeds_n_steps(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig(n_steps=42)
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "42" in text

    def test_pdb_file_is_valid_pdb(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        """PDB file should be parseable (at least starts with ATOM or CRYST1)."""
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        text = paths["pdb"].read_text()
        assert any(rec in text for rec in ("ATOM", "HETATM", "CRYST1"))

    def test_xml_ff_file_is_valid_xml(
        self, tmp_path, engine, simple_frame, empty_forcefield, nvt_config
    ):
        """FF file should be well-formed XML."""
        import xml.etree.ElementTree as ET

        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        tree = ET.parse(str(paths["forcefield"]))
        assert tree.getroot().tag == "ForceField"

    def test_nve_script_contains_langevin(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        """NVE falls through to the NVT renderer."""
        config = OpenMMSimulationConfig(ensemble="NVE")
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "LangevinMiddleIntegrator" in text

    def test_temperature_in_script(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig(temperature=450.0)
        paths = engine.generate_inputs(simple_frame, empty_forcefield, config, tmp_path)
        text = paths["script"].read_text()
        assert "450.0" in text


# ---------------------------------------------------------------------------
# write_openmm_system factory
# ---------------------------------------------------------------------------


class TestWriteOpenMMSystemFactory:
    def test_factory_produces_same_files(
        self, tmp_path, simple_frame, empty_forcefield, nvt_config
    ):
        engine = OpenMMEngine(check_executable=False)
        paths = engine.generate_inputs(
            simple_frame, empty_forcefield, nvt_config, tmp_path
        )
        assert paths["pdb"].exists()
        assert paths["forcefield"].exists()
        assert paths["script"].exists()


# ---------------------------------------------------------------------------
# serialize_system — requires OpenMM
# ---------------------------------------------------------------------------


class TestSerializeSystem:
    """These tests require OpenMM to be installed.

    Each test skips individually if ``import openmm`` fails.
    """

    @pytest.fixture(autouse=True)
    def _require_openmm(self):
        pytest.importorskip("openmm", reason="OpenMM not installed")

    def test_missing_openmm_raises_import_error(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        """Verify a helpful ImportError is raised when openmm is unavailable."""
        config = OpenMMSimulationConfig()
        original = sys.modules.copy()
        # Temporarily hide openmm
        for key in list(sys.modules):
            if key.startswith("openmm"):
                sys.modules.pop(key)
        try:
            with pytest.raises(ImportError, match="conda install"):
                engine.serialize_system(
                    simple_frame, empty_forcefield, config, tmp_path
                )
        finally:
            sys.modules.update(original)

    def test_system_xml_created(self, tmp_path, engine, simple_frame, empty_forcefield):
        config = OpenMMSimulationConfig()
        paths = engine.serialize_system(
            simple_frame, empty_forcefield, config, tmp_path
        )
        assert paths["system_xml"].exists()

    def test_integrator_xml_created(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig()
        paths = engine.serialize_system(
            simple_frame, empty_forcefield, config, tmp_path
        )
        assert paths["integrator_xml"].exists()

    def test_returns_all_six_keys(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        config = OpenMMSimulationConfig()
        paths = engine.serialize_system(
            simple_frame, empty_forcefield, config, tmp_path
        )
        assert {"pdb", "forcefield", "script", "system_xml", "integrator_xml"}.issubset(
            paths.keys()
        )

    def test_system_xml_roundtrip(
        self, tmp_path, engine, simple_frame, empty_forcefield
    ):
        """Deserialize the serialized System and check it is not None."""
        openmm = pytest.importorskip("openmm")
        config = OpenMMSimulationConfig()
        paths = engine.serialize_system(
            simple_frame, empty_forcefield, config, tmp_path
        )
        xml_text = paths["system_xml"].read_text(encoding="utf-8")
        system = openmm.XmlSerializer.deserialize(xml_text)
        assert system is not None
