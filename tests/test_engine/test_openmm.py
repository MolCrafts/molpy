"""Tests for OpenmmEngine and OpenmmSimulationConfig."""

import json
import sys
import types

import numpy as np
import pytest

from molpy.config import load_config
from molpy.core import Block, Frame
from molpy.engine import OpenmmEngine, OpenmmSimulationConfig
from molpy.ff.forcefield import ForceField

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
    return OpenmmSimulationConfig(ensemble="NVT", n_steps=100)


@pytest.fixture
def engine():
    return OpenmmEngine(check_executable=False)


# ---------------------------------------------------------------------------
# OpenmmSimulationConfig
# ---------------------------------------------------------------------------


class TestOpenmmSimulationConfig:
    def test_default_values(self):
        cfg = OpenmmSimulationConfig()
        assert cfg.ensemble == "NVT"
        assert cfg.temperature == 300.0
        assert cfg.timestep_fs == 2.0
        assert cfg.n_steps == 500_000
        assert cfg.nonbonded_method == "PME"
        assert cfg.constraints == "HBonds"
        assert cfg.platform == "CUDA"

    def test_npt_ensemble(self):
        cfg = OpenmmSimulationConfig(ensemble="NPT", pressure=2.0)
        assert cfg.ensemble == "NPT"
        assert cfg.pressure == 2.0

    def test_minimize_ensemble(self):
        cfg = OpenmmSimulationConfig(ensemble="minimize")
        assert cfg.ensemble == "minimize"

    def test_to_dict_returns_dict(self):
        cfg = OpenmmSimulationConfig(n_steps=1000)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["n_steps"] == 1000
        assert d["ensemble"] == "NVT"

    def test_to_dict_roundtrip(self):
        cfg = OpenmmSimulationConfig(temperature=350.0, n_steps=2000)
        restored = OpenmmSimulationConfig.from_dict(cfg.to_dict())
        assert restored.temperature == 350.0
        assert restored.n_steps == 2000

    def test_from_dict(self):
        d = {"ensemble": "NPT", "temperature": 400.0, "n_steps": 5000}
        cfg = OpenmmSimulationConfig.from_dict(d)
        assert cfg.ensemble == "NPT"
        assert cfg.temperature == 400.0

    def test_to_json_creates_file(self, tmp_path):
        cfg = OpenmmSimulationConfig(n_steps=999)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        assert path.exists()

    def test_to_json_roundtrip(self, tmp_path):
        cfg = OpenmmSimulationConfig(temperature=280.0, n_steps=1234)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        restored = OpenmmSimulationConfig.from_json(path)
        assert restored.temperature == 280.0
        assert restored.n_steps == 1234

    def test_json_file_is_valid_json(self, tmp_path):
        cfg = OpenmmSimulationConfig()
        path = tmp_path / "config.json"
        cfg.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ensemble" in data
        assert "temperature" in data

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("ensemble", "nvt"),
            ("ensemble", "NVX"),
            ("nonbonded_method", "EWALD"),
            ("nonbonded_method", "pme"),
            ("constraints", "hbonds"),
            ("constraints", None),
        ],
    )
    def test_rejects_unknown_choice(self, field, value):
        with pytest.raises(ValueError, match=f"{field} must be one of"):
            OpenmmSimulationConfig(**{field: value})

    def test_from_dict_rejects_unknown_ensemble(self):
        with pytest.raises(ValueError, match="ensemble must be one of"):
            OpenmmSimulationConfig.from_dict({"ensemble": "nve"})

    def test_from_json_rejects_unknown_ensemble(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ensemble": "nvt"}), encoding="utf-8")
        with pytest.raises(ValueError, match="ensemble must be one of"):
            OpenmmSimulationConfig.from_json(path)

    def test_accepts_openmm_ewald_name(self):
        cfg = OpenmmSimulationConfig(nonbonded_method="Ewald")
        assert cfg.nonbonded_method == "Ewald"


# ---------------------------------------------------------------------------
# OpenmmEngine initialisation
# ---------------------------------------------------------------------------


class TestOpenmmEngineInit:
    def test_key(self, engine):
        assert engine.key == "engine.openmm"

    def test_extension(self, engine):
        assert engine._get_default_extension() == ".py"

    def test_default_executable(self, engine):
        assert engine.executable == "python"

    def test_check_executable_false_does_not_raise(self):
        config = load_config({"engine": {"openmm": {"executable": "no_such_xyz"}}})
        OpenmmEngine(config=config, check_executable=False)


def test_no_serialize_system(engine):
    """Writing files is molrs io's job: the engine only renders the script."""
    assert not hasattr(engine, "serialize_system")


# ---------------------------------------------------------------------------
# The rendered script — text, and a run against a stand-in openmm package
# ---------------------------------------------------------------------------


def _script(engine, **config_fields):
    return engine._render_simulation_script(
        OpenmmSimulationConfig(**config_fields),
        pdb_filename="system.pdb",
        ff_filename="forcefield.xml",
    )


class TestRenderSimulationScript:
    @pytest.mark.parametrize("ensemble", ["NVT", "NPT", "NVE", "minimize"])
    def test_script_compiles(self, engine, ensemble):
        compile(_script(engine, ensemble=ensemble), "simulate.py", "exec")

    def test_nve_uses_verlet_integrator(self, engine):
        script = _script(engine, ensemble="NVE")
        assert "integrator = VerletIntegrator(" in script
        assert "LangevinMiddleIntegrator" not in script
        assert "MonteCarloBarostat" not in script
        assert "setVelocitiesToTemperature(300.0*unit.kelvin)" in script
        assert "Running NVE" in script

    def test_nvt_uses_langevin_without_barostat(self, engine):
        script = _script(engine, ensemble="NVT")
        assert "integrator = LangevinMiddleIntegrator(" in script
        assert "MonteCarloBarostat" not in script
        assert "setVelocitiesToTemperature" not in script
        assert "Running NVT" in script

    def test_npt_adds_barostat(self, engine):
        script = _script(engine, ensemble="NPT", pressure=2.0)
        assert "system.addForce(MonteCarloBarostat(" in script
        assert "2.0*unit.bar" in script
        assert "Running NPT" in script

    def test_ewald_imported_by_openmm_name(self, engine):
        script = _script(engine, nonbonded_method="Ewald")
        assert "    Ewald,\n" in script
        assert "nonbondedMethod=Ewald," in script

    def test_minimize_writes_the_minimized_coordinates(self, engine):
        script = _script(engine, ensemble="minimize", output_pdb="relaxed.pdb")
        assert "PDBFile.writeFile(" in script
        assert "'relaxed.pdb'" in script
        assert "simulation.step(" not in script


class _Unit:
    def __init__(self, name):
        self.name = name

    def __rmul__(self, value):
        return (value, self.name)

    def __rtruediv__(self, value):
        return (value, f"1/{self.name}")


class _Call:
    """Records the arguments it was constructed with."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.fixture
def fake_openmm(monkeypatch):
    """A minimal ``openmm`` package; returns the record of what a script did."""
    seen = {"forces": [], "velocities": [], "written": [], "steps": []}

    class System:
        def addForce(self, force):
            seen["forces"].append(force)

    class ForceField(_Call):
        def createSystem(self, *args, **kwargs):
            seen["createSystem"] = kwargs
            return System()

    class PDBFile(_Call):
        topology = "topology"
        positions = "initial positions"

        @staticmethod
        def writeFile(topology, positions, handle):
            seen["written"].append((topology, positions, handle.name))

    class Context:
        def setPositions(self, positions):
            seen["positions"] = positions

        def setVelocitiesToTemperature(self, temperature):
            seen["velocities"].append(temperature)

        def getState(self, **kwargs):
            return types.SimpleNamespace(getPositions=lambda: "minimized positions")

    class Simulation(_Call):
        topology = "topology"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            seen["integrator"] = args[2]
            self.context = Context()
            self.reporters = []

        def minimizeEnergy(self, **kwargs):
            seen["minimized"] = kwargs

        def step(self, n):
            seen["steps"].append(n)

    unit = types.ModuleType("openmm.unit")
    for name in ("nanometer", "picosecond", "femtoseconds", "kelvin", "bar"):
        setattr(unit, name, _Unit(name))
    unit.kilojoule, unit.mole = 1.0, 1.0
    app = types.ModuleType("openmm.app")
    app.PDBFile, app.ForceField, app.Simulation = PDBFile, ForceField, Simulation
    for name in ("DCDReporter", "StateDataReporter", "CheckpointReporter"):
        setattr(app, name, _Call)
    for name in (
        *("NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald"),
        *("HBonds", "AllBonds", "HAngles"),
    ):
        setattr(app, name, f"app.{name}")
    openmm = types.ModuleType("openmm")
    openmm.app, openmm.unit = app, unit
    openmm.Platform = types.SimpleNamespace(getPlatformByName=lambda name: name)
    for name in ("LangevinMiddleIntegrator", "VerletIntegrator", "MonteCarloBarostat"):
        setattr(openmm, name, type(name, (_Call,), {}))
    monkeypatch.setitem(sys.modules, "openmm", openmm)
    monkeypatch.setitem(sys.modules, "openmm.app", app)
    monkeypatch.setitem(sys.modules, "openmm.unit", unit)
    return seen


def _run_script(engine, tmp_path, monkeypatch, **config_fields):
    monkeypatch.chdir(tmp_path)
    exec(compile(_script(engine, **config_fields), "simulate.py", "exec"), {})


class TestScriptRun:
    def test_nve_integrates_with_verlet_from_thermal_velocities(
        self, engine, fake_openmm, tmp_path, monkeypatch
    ):
        _run_script(engine, tmp_path, monkeypatch, ensemble="NVE", temperature=250.0)
        assert type(fake_openmm["integrator"]).__name__ == "VerletIntegrator"
        assert fake_openmm["integrator"].args == ((2.0, "femtoseconds"),)
        assert fake_openmm["velocities"] == [(250.0, "kelvin")]
        assert fake_openmm["forces"] == []

    def test_npt_system_carries_the_barostat(
        self, engine, fake_openmm, tmp_path, monkeypatch
    ):
        _run_script(engine, tmp_path, monkeypatch, ensemble="NPT", pressure=2.0)
        (barostat,) = fake_openmm["forces"]
        assert barostat.args == ((2.0, "bar"), (300.0, "kelvin"), 25)
        assert type(fake_openmm["integrator"]).__name__ == "LangevinMiddleIntegrator"

    def test_system_uses_configured_nonbonded_and_constraints(
        self, engine, fake_openmm, tmp_path, monkeypatch
    ):
        _run_script(
            engine,
            tmp_path,
            monkeypatch,
            nonbonded_method="Ewald",
            constraints="None",
            nonbonded_cutoff_nm=0.9,
        )
        assert fake_openmm["createSystem"] == {
            "nonbondedMethod": "app.Ewald",
            "nonbondedCutoff": (0.9, "nanometer"),
            "constraints": None,
        }

    def test_minimize_writes_its_coordinates(
        self, engine, fake_openmm, tmp_path, monkeypatch
    ):
        _run_script(engine, tmp_path, monkeypatch, ensemble="minimize")
        assert fake_openmm["written"] == [
            ("topology", "minimized positions", "minimized.pdb")
        ]
        assert fake_openmm["steps"] == []
