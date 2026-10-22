"""Tests for OpenmmEngine and OpenmmSimulationConfig."""

import json
import sys
import types

import pytest

import molpy as mp

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return mp.engine.OpenmmEngine(check_executable=False)


# ---------------------------------------------------------------------------
# OpenmmSimulationConfig
# ---------------------------------------------------------------------------


class TestOpenmmSimulationConfig:
    def test_default_values(self):
        cfg = mp.engine.OpenmmSimulationConfig()
        assert cfg.ensemble == "NVT"
        assert cfg.temperature == 300.0
        assert cfg.timestep_fs == 2.0
        assert cfg.n_steps == 500_000
        assert cfg.nonbonded_method is None  # the force field's
        assert cfg.nonbonded_cutoff_nm is None
        assert cfg.constraints == "HBonds"
        assert cfg.platform == "CUDA"

    def test_npt_ensemble(self):
        cfg = mp.engine.OpenmmSimulationConfig(ensemble="NPT", pressure=2.0)
        assert cfg.ensemble == "NPT"
        assert cfg.pressure == 2.0

    def test_minimize_ensemble(self):
        cfg = mp.engine.OpenmmSimulationConfig(ensemble="minimize")
        assert cfg.ensemble == "minimize"

    def test_to_dict_returns_dict(self):
        cfg = mp.engine.OpenmmSimulationConfig(n_steps=1000)
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["n_steps"] == 1000
        assert d["ensemble"] == "NVT"

    def test_to_dict_roundtrip(self):
        cfg = mp.engine.OpenmmSimulationConfig(temperature=350.0, n_steps=2000)
        restored = mp.engine.OpenmmSimulationConfig.from_dict(cfg.to_dict())
        assert restored.temperature == 350.0
        assert restored.n_steps == 2000

    def test_from_dict(self):
        d = {"ensemble": "NPT", "temperature": 400.0, "n_steps": 5000}
        cfg = mp.engine.OpenmmSimulationConfig.from_dict(d)
        assert cfg.ensemble == "NPT"
        assert cfg.temperature == 400.0

    def test_to_json_creates_file(self, tmp_path):
        cfg = mp.engine.OpenmmSimulationConfig(n_steps=999)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        assert path.exists()

    def test_to_json_roundtrip(self, tmp_path):
        cfg = mp.engine.OpenmmSimulationConfig(temperature=280.0, n_steps=1234)
        path = tmp_path / "config.json"
        cfg.to_json(path)
        restored = mp.engine.OpenmmSimulationConfig.from_json(path)
        assert restored.temperature == 280.0
        assert restored.n_steps == 1234

    def test_json_file_is_valid_json(self, tmp_path):
        cfg = mp.engine.OpenmmSimulationConfig()
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
            mp.engine.OpenmmSimulationConfig(**{field: value})

    def test_from_dict_rejects_unknown_ensemble(self):
        with pytest.raises(ValueError, match="ensemble must be one of"):
            mp.engine.OpenmmSimulationConfig.from_dict({"ensemble": "nve"})

    def test_from_json_rejects_unknown_ensemble(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"ensemble": "nvt"}), encoding="utf-8")
        with pytest.raises(ValueError, match="ensemble must be one of"):
            mp.engine.OpenmmSimulationConfig.from_json(path)

    def test_accepts_openmm_ewald_name(self):
        cfg = mp.engine.OpenmmSimulationConfig(nonbonded_method="Ewald")
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
        config = mp.config.load_config(
            {"engine": {"openmm": {"executable": "no_such_xyz"}}}
        )
        mp.engine.OpenmmEngine(config=config, check_executable=False)


# ---------------------------------------------------------------------------
# generate_inputs — no OpenMM required
# ---------------------------------------------------------------------------


@pytest.fixture
def water_system(water, tip3p):
    """The water and a force field OpenMM can price: charges need Coulomb."""
    tip3p.def_style("pair", "coul/cut")
    return water.to_frame(), tip3p


def _pdb_atom_names(path):
    return [
        line[12:16].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith(("ATOM", "HETATM"))
    ]


class TestGenerateInputs:
    def test_writes_pdb_forcefield_and_script(self, engine, water_system, tmp_path):
        paths = engine.generate_inputs(*water_system, tmp_path)
        assert {key: p.name for key, p in paths.items()} == {
            "input": "simulate.py",
            "pdb": "system.pdb",
            "forcefield": "forcefield.xml",
        }
        assert all(p.is_file() for p in paths.values())

    def test_forcefield_carries_the_pdb_residue_templates(
        self, engine, water_system, tmp_path
    ):
        """createSystem matches every PDB residue against a <Residue>
        template by its atom names: the XML has one per residue, named as the
        PDB names its atoms."""
        import xml.etree.ElementTree as ET

        paths = engine.generate_inputs(*water_system, tmp_path)
        root = ET.parse(paths["forcefield"]).getroot()
        residues = root.find("Residues")
        assert residues is not None
        (template,) = residues.findall("Residue")
        assert [atom.get("name") for atom in template.findall("Atom")] == (
            _pdb_atom_names(paths["pdb"])
        )
        assert [atom.get("type") for atom in template.findall("Atom")] == [
            "OW",
            "HW",
            "HW",
        ]


class TestNonbondedFromTheForceField:
    """Unset nonbonded fields are the force field's; set ones override it."""

    @staticmethod
    def _create_system(engine, frame, ff, tmp_path, **config_fields):
        paths = engine.generate_inputs(
            frame,
            ff,
            tmp_path,
            simulation=mp.engine.OpenmmSimulationConfig(**config_fields),
        )
        script = paths["input"].read_text(encoding="utf-8")
        start = script.index("system = ff.createSystem(")
        return script[start : script.index(")\n", start)]

    @pytest.fixture
    def boxed_water(self, water, tip3p):
        tip3p.get_style("pair", "lj/cut")["cutoff"] = 9.0
        frame = water.to_frame()
        frame.box = mp.Box(mp.core.Cell.lengths([30.0] * 3))
        return frame, tip3p

    def test_long_range_coulomb_is_pme_at_the_force_field_cutoff(self, boxed_water):
        """molrs writes no OpenMM XML for ``coul/long/pme`` yet, so this is
        the resolution alone."""
        from molpy.engine.openmm import _with_nonbonded

        _, ff = boxed_water
        ff.def_style("pair", "coul/long/pme", {"cutoff": 12.0})
        config = _with_nonbonded(mp.engine.OpenmmSimulationConfig(), ff, periodic=True)
        assert config.nonbonded_method == "PME"
        assert config.nonbonded_cutoff_nm == pytest.approx(1.2)

    def test_no_stated_settings_fall_back_to_pme_at_1_nm(self, boxed_water):
        from molpy.engine.openmm import _with_nonbonded

        config = _with_nonbonded(
            mp.engine.OpenmmSimulationConfig(),
            mp.ff.forcefield.ForceField("bare"),
            periodic=True,
        )
        assert config.nonbonded_method == "PME"
        assert config.nonbonded_cutoff_nm == 1.0

    def test_cut_coulomb_is_a_periodic_cutoff(self, engine, boxed_water, tmp_path):
        frame, ff = boxed_water
        ff.def_style("pair", "coul/cut")
        setup = self._create_system(engine, frame, ff, tmp_path)
        assert "nonbondedMethod=CutoffPeriodic," in setup
        assert "nonbondedCutoff=0.9*unit.nanometer," in setup

    def test_explicit_settings_override_the_force_field(
        self, engine, boxed_water, tmp_path
    ):
        frame, ff = boxed_water
        ff.def_style("pair", "coul/cut")
        setup = self._create_system(
            engine,
            frame,
            ff,
            tmp_path,
            nonbonded_method="Ewald",
            nonbonded_cutoff_nm=0.8,
        )
        assert "nonbondedMethod=Ewald," in setup
        assert "nonbondedCutoff=0.8*unit.nanometer," in setup


class TestBoxFreeFrame:
    """A frame without a box is a non-periodic system: NoCutoff by default,
    and a method needing a periodic cell is refused before any file."""

    def test_defaults_to_no_cutoff(self, engine, water_system, tmp_path):
        paths = engine.generate_inputs(*water_system, tmp_path)
        script = paths["input"].read_text(encoding="utf-8")
        assert "nonbondedMethod=NoCutoff," in script

    def test_a_non_periodic_cutoff_is_kept(self, engine, water_system, tmp_path):
        config = mp.engine.OpenmmSimulationConfig(nonbonded_method="CutoffNonPeriodic")
        paths = engine.generate_inputs(*water_system, tmp_path, simulation=config)
        assert "nonbondedMethod=CutoffNonPeriodic," in paths["input"].read_text(
            encoding="utf-8"
        )

    @pytest.mark.parametrize(
        ("fields", "match"),
        [
            ({"nonbonded_method": "PME"}, "nonbonded_method 'PME'"),
            ({"nonbonded_method": "Ewald"}, "nonbonded_method 'Ewald'"),
            ({"nonbonded_method": "CutoffPeriodic"}, "CutoffPeriodic"),
            ({"ensemble": "NPT"}, "NPT"),
        ],
    )
    def test_periodic_methods_are_refused(
        self, engine, water_system, tmp_path, fields, match
    ):
        with pytest.raises(ValueError, match=f"{match}.*needs a periodic box"):
            engine.generate_inputs(
                *water_system,
                tmp_path / "out",
                simulation=mp.engine.OpenmmSimulationConfig(**fields),
            )
        assert not (tmp_path / "out").exists()


def test_no_serialize_system(engine):
    """Writing files is molrs io's job: the engine only renders the script."""
    assert not hasattr(engine, "serialize_system")


# ---------------------------------------------------------------------------
# The rendered script — text, and a run against a stand-in openmm package
# ---------------------------------------------------------------------------


def _script(engine, **config_fields):
    config_fields = {"nonbonded_method": "PME", "nonbonded_cutoff_nm": 1.0} | (
        config_fields
    )
    return engine._render_simulation_script(
        mp.engine.OpenmmSimulationConfig(**config_fields),
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
