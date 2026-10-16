"""OpenMM simulation engine for MolPy.

Generates OpenMM input files (PDB + XML force field + Python simulation
script) from :class:`~molpy.Frame` and
:class:`~molpy.ff.forcefield.ForceField` objects.  OpenMM itself is **not**
required for input generation; it is only needed to run the script.

Two usage modes are supported:

1. **Generate only** (no OpenMM required)::

       config = OpenmmSimulationConfig(ensemble="NVT", n_steps=50_000)
       engine = OpenmmEngine(check_executable=False)
       paths = engine.generate_inputs(frame, ff, config, "./output")
       # Hand the files to any HPC scheduler.

2. **Generate and run** (OpenMM must be importable)::

       result = engine.run(paths["script"], workdir="./output")

The Python interpreter that runs the script, its environment, launcher and
time limit are the ``[engine.openmm]`` settings of molpy's configuration
(:mod:`molpy.config`); runs are logged to ``molpy.engine.openmm``.

Reference:
    Eastman, P. et al. (2017). OpenMM 7: Rapid development of high
    performance algorithms for molecular dynamics. *PLOS Comput. Biol.*
    **13**(7), e1005659.
    https://doi.org/10.1371/journal.pcbi.1005659
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from molpy.engine._engine import Engine
from molpy.io import write_openmm_system, write_pdb

if TYPE_CHECKING:
    from molrs.core import Frame
    from molrs.ff.forcefield import ForceField

PathLike = str | Path

Ensemble = Literal["NVT", "NPT", "NVE", "minimize"]
NonbondedMethod = Literal[
    "NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald"
]
Constraints = Literal["None", "HBonds", "AllBonds", "HAngles"]


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OpenmmSimulationConfig:
    """Configuration for an OpenMM simulation.

    All fields have physical units encoded in their names.  The object
    round-trips through JSON via :meth:`to_json` / :meth:`from_json`.

    Attributes:
        ensemble: Simulation ensemble (case-sensitive). ``"NVT"`` and
            ``"NPT"`` use a ``LangevinMiddleIntegrator`` (NPT adds a
            ``MonteCarloBarostat``); ``"NVE"`` uses a ``VerletIntegrator``
            and starts from velocities drawn at *temperature*;
            ``"minimize"`` only minimises the energy and writes the
            minimised coordinates to *output_pdb*.
        temperature: Temperature in Kelvin (the thermostat's, or NVE's
            initial velocities').
        pressure: Pressure in bar (NPT only).
        timestep_fs: Integration timestep in femtoseconds.
        n_steps: Number of MD steps.
        nonbonded_method: Nonbonded treatment, by its ``openmm.app`` name:
            ``"NoCutoff"``, ``"CutoffNonPeriodic"``, ``"CutoffPeriodic"``,
            ``"PME"`` or ``"Ewald"``.
        nonbonded_cutoff_nm: Nonbonded cutoff radius in nanometres.
        constraints: Constraint scheme — ``"HBonds"``, ``"AllBonds"``,
            ``"HAngles"``, or the string ``"None"`` for no constraints.
        friction_per_ps: Langevin friction coefficient in ps⁻¹ (NVT, NPT).
        dcd_reporter_interval: Steps between DCD trajectory frames.
        state_reporter_interval: Steps between log lines.
        checkpoint_interval: Steps between checkpoint saves.
        output_dcd: Filename for the DCD trajectory.
        output_log: Filename for the state-data log.
        output_chk: Filename for checkpoint files.
        output_pdb: Filename for the minimised coordinates (``"minimize"``).
        minimize_tolerance: Energy minimisation tolerance in kJ mol⁻¹ nm⁻¹.
        minimize_max_iterations: Maximum minimisation iterations.
        barostat_frequency: MC barostat move frequency in steps (NPT only).
        platform: OpenMM platform name (``"CUDA"``, ``"OpenCL"``, ``"CPU"``).

    Raises:
        ValueError: If *ensemble*, *nonbonded_method* or *constraints* is not
            one of the values listed above (also from :meth:`from_dict` and
            :meth:`from_json`).
    """

    ensemble: Ensemble = "NVT"
    temperature: float = 300.0
    pressure: float = 1.0
    timestep_fs: float = 2.0
    n_steps: int = 500_000
    nonbonded_method: NonbondedMethod = "PME"
    nonbonded_cutoff_nm: float = 1.0
    constraints: Constraints = "HBonds"
    friction_per_ps: float = 1.0
    dcd_reporter_interval: int = 1000
    state_reporter_interval: int = 1000
    checkpoint_interval: int = 10_000
    output_dcd: str = "trajectory.dcd"
    output_log: str = "simulation.log"
    output_chk: str = "checkpoint.chk"
    output_pdb: str = "minimized.pdb"
    minimize_tolerance: float = 10.0
    minimize_max_iterations: int = 1000
    barostat_frequency: int = 25
    platform: str = "CUDA"

    def __post_init__(self) -> None:
        for field, choices in (
            ("ensemble", get_args(Ensemble)),
            ("nonbonded_method", get_args(NonbondedMethod)),
            ("constraints", get_args(Constraints)),
        ):
            value = getattr(self, field)
            if value not in choices:
                raise ValueError(
                    f"{field} must be one of {list(choices)}, got {value!r}."
                )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary of all fields.

        Returns:
            Plain dict with field names as keys and Python scalars as values.
        """
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OpenmmSimulationConfig":
        """Construct a config from a plain dictionary.

        Args:
            data: Dict as returned by :meth:`to_dict`.

        Returns:
            New :class:`OpenmmSimulationConfig` instance.

        Raises:
            ValueError: If a choice field holds an unknown value.
        """
        return cls(**data)

    def to_json(self, path: PathLike) -> None:
        """Write the configuration to a JSON file.

        Args:
            path: Destination file path (created or overwritten).

        Raises:
            OSError: If the file cannot be written.
        """
        Path(path).write_text(
            json.dumps(dataclasses.asdict(self), indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: PathLike) -> "OpenmmSimulationConfig":
        """Load a configuration from a JSON file.

        Args:
            path: Path to a JSON file previously written by :meth:`to_json`.

        Returns:
            New :class:`OpenmmSimulationConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If a choice field holds an unknown value.
        """
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OpenmmEngine(Engine):
    """OpenMM molecular dynamics engine.

    Generates a complete set of OpenMM input files from MolPy
    :class:`~molpy.Frame` and :class:`~molpy.ff.forcefield.ForceField`
    objects.  OpenMM itself is **not** required for input generation; it is
    only needed to run the generated script.

    The generated Python script can be executed directly::

        python simulate.py

    or via the engine::

        result = engine.run(paths["script"], workdir="./output")

    which runs ``engine.openmm.executable`` (the Python interpreter, ``python``
    by default) in the configured environment.  Settings:
    ``[engine.openmm]``; logger: ``molpy.engine.openmm``.

    Example:
        >>> from molpy.engine import OpenmmEngine, OpenmmSimulationConfig
        >>>
        >>> config = OpenmmSimulationConfig(ensemble="NVT", n_steps=10_000)
        >>> engine = OpenmmEngine(check_executable=False)
        >>> paths = engine.generate_inputs(frame, ff, config, "./output")
        >>> # paths["pdb"], paths["forcefield"], paths["script"]
    """

    tool = "openmm"

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_default_extension(self) -> str:
        """Return ``".py"`` — the extension for generated simulation scripts.

        Returns:
            ``".py"``
        """
        return ".py"

    def _execute(
        self,
        run_dir: Path,
        input_file: Path,
        *,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Run the Python simulation script *input_file* in *run_dir*.

        Builds the command::

            [launcher...] python simulate.py

        OpenMM must be importable in the configured Python interpreter.

        Raises:
            ToolError: The script could not start, timed out, or (with
                *check*) exited non-zero.
        """
        return self.run_command(
            self.command([input_file.name]),
            cwd=run_dir,
            capture_output=capture_output,
            check=check,
        )

    # ------------------------------------------------------------------
    # Public input-generation API
    # ------------------------------------------------------------------

    def generate_inputs(
        self,
        frame: "Frame",
        forcefield: "ForceField",
        config: OpenmmSimulationConfig,
        output_dir: PathLike,
        *,
        pdb_filename: str = "system.pdb",
        ff_filename: str = "forcefield.xml",
        script_filename: str = "simulate.py",
    ) -> dict[str, Path]:
        """Generate PDB, XML force field, and Python simulation script.

        Does **not** require OpenMM to be installed. The PDB is molrs's
        :func:`~molpy.io.write_pdb`; the force-field XML is
        :func:`~molpy.io.write_openmm_system`: the force field plus a
        ``<Residue>`` template for every residue of that PDB (its atom names,
        types, charges and bonds), what ``ForceField.createSystem`` matches
        the PDB topology against.

        Args:
            frame: Typed :class:`~molpy.Frame` (string ``type`` and
                ``element`` atom columns, ``bonds``) with atom positions.
            forcefield: A :class:`~molpy.ff.forcefield.ForceField` holding an
                atom type for every ``type`` the frame names.
            config: Simulation parameters.
            output_dir: Directory where files are written (created if absent).
            pdb_filename: Name for the PDB coordinate file.
            ff_filename: Name for the XML force field file.
            script_filename: Name for the Python simulation script.

        Returns:
            Dictionary with keys ``"pdb"``, ``"forcefield"``, and ``"script"``
            mapping to :class:`~pathlib.Path` objects.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pdb_path = out / pdb_filename
        ff_path = out / ff_filename
        script_path = out / script_filename

        # The XML carries a residue template for every residue of the PDB
        # (same atom names), which createSystem matches the topology against.
        write_pdb(pdb_path, frame)
        write_openmm_system(ff_path, forcefield, frame)

        script_text = self._render_simulation_script(
            config, pdb_filename=pdb_filename, ff_filename=ff_filename
        )
        script_path.write_text(script_text, encoding="utf-8")

        paths = {"pdb": pdb_path, "forcefield": ff_path, "script": script_path}
        self.logger.info(
            "inputs written",
            output_dir=str(out),
            files={key: path.name for key, path in paths.items()},
            ensemble=config.ensemble,
            n_steps=config.n_steps,
        )
        return paths

    # ------------------------------------------------------------------
    # Private script rendering
    # ------------------------------------------------------------------

    def _render_simulation_script(
        self,
        config: OpenmmSimulationConfig,
        *,
        pdb_filename: str,
        ff_filename: str,
    ) -> str:
        """Render a self-contained Python simulation script.

        The setup (imports, ``system``, the NPT barostat, the integrator and
        the ``Simulation``) is followed by the run: a minimisation that
        writes its coordinates, or a minimisation then the MD steps. Uses
        f-strings for value substitution so that OpenMM unit expressions such
        as ``300*unit.kelvin`` are never misinterpreted as format tokens.

        Args:
            config: Simulation parameters.
            pdb_filename: PDB filename embedded in the script.
            ff_filename: Force field XML filename embedded in the script.

        Returns:
            Complete Python script as a string.
        """
        run = (
            self._minimize_run(config)
            if config.ensemble == "minimize"
            else self._md_run(config)
        )
        return (
            self._import_block(config)
            + self._system_setup(config, pdb_filename, ff_filename)
            + self._barostat(config)
            + self._integrator(config)
            + self._simulation(config)
            + run
        )

    @staticmethod
    def _import_block(config: OpenmmSimulationConfig) -> str:
        """The script's docstring and imports, the same for every ensemble."""
        constraints_import = (
            f"    {config.constraints},\n" if config.constraints != "None" else ""
        )
        return (
            '"""OpenMM simulation script — generated by MolPy."""\n'
            "from openmm.app import (\n"
            "    PDBFile,\n"
            "    ForceField,\n"
            "    Simulation,\n"
            "    DCDReporter,\n"
            "    StateDataReporter,\n"
            "    CheckpointReporter,\n"
            f"    {config.nonbonded_method},\n"
            f"{constraints_import}"
            ")\n"
            "from openmm import (\n"
            "    Platform,\n"
            "    unit,\n"
            ")\n"
        )

    @staticmethod
    def _system_setup(
        config: OpenmmSimulationConfig, pdb_filename: str, ff_filename: str
    ) -> str:
        """Render PDB loading and system creation lines."""
        return (
            f"\npdb = PDBFile({pdb_filename!r})\n"
            f"ff  = ForceField({ff_filename!r})\n"
            "system = ff.createSystem(\n"
            "    pdb.topology,\n"
            f"    nonbondedMethod={config.nonbonded_method},\n"
            f"    nonbondedCutoff={config.nonbonded_cutoff_nm}*unit.nanometer,\n"
            f"    constraints={config.constraints},\n"
            ")\n"
        )

    @staticmethod
    def _barostat(config: OpenmmSimulationConfig) -> str:
        """Render the NPT Monte Carlo barostat; other ensembles have none."""
        if config.ensemble != "NPT":
            return ""
        return (
            "from openmm import MonteCarloBarostat\n"
            "system.addForce(MonteCarloBarostat(\n"
            f"    {config.pressure}*unit.bar,\n"
            f"    {config.temperature}*unit.kelvin,\n"
            f"    {config.barostat_frequency},\n"
            "))\n"
        )

    @staticmethod
    def _integrator(config: OpenmmSimulationConfig) -> str:
        """Render the integrator: Verlet for NVE, Langevin otherwise."""
        if config.ensemble == "NVE":
            return (
                "from openmm import VerletIntegrator\n"
                f"integrator = VerletIntegrator({config.timestep_fs}*unit.femtoseconds)\n"
            )
        return (
            "from openmm import LangevinMiddleIntegrator\n"
            "integrator = LangevinMiddleIntegrator(\n"
            f"    {config.temperature}*unit.kelvin,\n"
            f"    {config.friction_per_ps}/unit.picosecond,\n"
            f"    {config.timestep_fs}*unit.femtoseconds,\n"
            ")\n"
        )

    @staticmethod
    def _simulation(config: OpenmmSimulationConfig) -> str:
        """Render the Simulation on the configured platform."""
        return (
            f"platform = Platform.getPlatformByName({config.platform!r})\n"
            "simulation = Simulation(pdb.topology, system, integrator, platform)\n"
            "simulation.context.setPositions(pdb.positions)\n"
        )

    @staticmethod
    def _reporters(config: OpenmmSimulationConfig) -> str:
        """Render reporter registration lines."""
        return (
            f"simulation.reporters.append(DCDReporter({config.output_dcd!r}, {config.dcd_reporter_interval}))\n"
            "simulation.reporters.append(StateDataReporter(\n"
            f"    {config.output_log!r}, {config.state_reporter_interval},\n"
            "    step=True, potentialEnergy=True, temperature=True, speed=True,\n"
            "))\n"
            f"simulation.reporters.append(CheckpointReporter({config.output_chk!r}, {config.checkpoint_interval}))\n"
        )

    def _md_run(self, config: OpenmmSimulationConfig) -> str:
        """Render minimisation, reporters and the MD steps (NVT, NPT, NVE)."""
        # Verlet has no thermostat to heat the system, so NVE starts from
        # velocities at the requested temperature rather than from rest.
        velocities = (
            "simulation.context.setVelocitiesToTemperature("
            f"{config.temperature}*unit.kelvin)\n"
            if config.ensemble == "NVE"
            else ""
        )
        return "".join(
            [
                "\nprint('Minimising energy...')\n",
                "simulation.minimizeEnergy()\n",
                velocities,
                "\n",
                self._reporters(config),
                f"\nprint('Running {config.ensemble} for {config.n_steps} steps...')\n",
                f"simulation.step({config.n_steps})\n",
                "print('Done.')\n",
            ]
        )

    @staticmethod
    def _minimize_run(config: OpenmmSimulationConfig) -> str:
        """Render the minimisation and the write of its coordinates."""
        return "".join(
            [
                "\nprint('Minimising energy...')\n",
                "simulation.minimizeEnergy(\n",
                f"    tolerance={config.minimize_tolerance}*unit.kilojoule/unit.mole/unit.nanometer,\n",
                f"    maxIterations={config.minimize_max_iterations},\n",
                ")\n",
                "state = simulation.context.getState(getPositions=True)\n",
                f"with open({config.output_pdb!r}, 'w') as handle:\n",
                "    PDBFile.writeFile(simulation.topology, state.getPositions(), handle)\n",
                "print('Done.')\n",
            ]
        )
