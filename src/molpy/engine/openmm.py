"""OpenMM simulation engine for MolPy.

Generates OpenMM input files (PDB + XML force field + Python simulation
script) from MolPy :class:`~molpy.core.Frame` and
:class:`~molpy.core.forcefield.ForceField` objects.  OpenMM itself is **not**
required for input generation; it is only needed for
:meth:`~OpenMMEngine.serialize_system`.

Two usage modes are supported:

1. **Generate only** (no OpenMM required)::

       config = OpenMMSimulationConfig(ensemble="NVT", n_steps=50_000)
       engine = OpenMMEngine(check_executable=False)
       paths = engine.generate_inputs(frame, ff, config, "./output")
       # Hand the files to any HPC scheduler.

2. **Generate and run** (OpenMM must be importable)::

       result = engine.run(paths["script"], workdir="./output")

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
from typing import TYPE_CHECKING, Any, Literal

from molpy.core.script import Script

from .base import Engine

if TYPE_CHECKING:
    from molpy.core import Frame
    from molpy.core.forcefield import ForceField

PathLike = str | Path


# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class OpenMMSimulationConfig:
    """Configuration for an OpenMM simulation.

    All fields have physical units encoded in their names.  The object
    round-trips through JSON via :meth:`to_json` / :meth:`from_json`.

    Attributes:
        ensemble: Simulation ensemble — ``"NVT"``, ``"NPT"``, ``"NVE"``, or
            ``"minimize"``.
        temperature: Temperature in Kelvin.
        pressure: Pressure in bar (NPT only).
        timestep_fs: Integration timestep in femtoseconds.
        n_steps: Number of MD steps.
        nonbonded_method: Nonbonded treatment.  Must be a valid
            ``openmm.app`` attribute name (``"NoCutoff"``,
            ``"CutoffNonPeriodic"``, ``"CutoffPeriodic"``, ``"PME"``,
            ``"EWALD"``).
        nonbonded_cutoff_nm: Nonbonded cutoff radius in nanometres.
        constraints: Constraint scheme — a valid ``openmm.app`` attribute
            name or ``"None"`` for no constraints.
        friction_per_ps: Langevin friction coefficient in ps⁻¹.
        dcd_reporter_interval: Steps between DCD trajectory frames.
        state_reporter_interval: Steps between log lines.
        checkpoint_interval: Steps between checkpoint saves.
        output_dcd: Filename for the DCD trajectory.
        output_log: Filename for the state-data log.
        output_chk: Filename for checkpoint files.
        minimize_tolerance: Energy minimisation tolerance in kJ mol⁻¹ nm⁻¹.
        minimize_max_iterations: Maximum minimisation iterations.
        barostat_frequency: MC barostat move frequency in steps (NPT only).
        platform: OpenMM platform name (``"CUDA"``, ``"OpenCL"``, ``"CPU"``).
    """

    ensemble: Literal["NVT", "NPT", "NVE", "minimize"] = "NVT"
    temperature: float = 300.0
    pressure: float = 1.0
    timestep_fs: float = 2.0
    n_steps: int = 500_000
    nonbonded_method: Literal[
        "NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "EWALD"
    ] = "PME"
    nonbonded_cutoff_nm: float = 1.0
    constraints: Literal["None", "HBonds", "AllBonds", "HAngles"] = "HBonds"
    friction_per_ps: float = 1.0
    dcd_reporter_interval: int = 1000
    state_reporter_interval: int = 1000
    checkpoint_interval: int = 10_000
    output_dcd: str = "trajectory.dcd"
    output_log: str = "simulation.log"
    output_chk: str = "checkpoint.chk"
    minimize_tolerance: float = 10.0
    minimize_max_iterations: int = 1000
    barostat_frequency: int = 25
    platform: str = "CUDA"

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
    def from_dict(cls, data: dict[str, Any]) -> "OpenMMSimulationConfig":
        """Construct a config from a plain dictionary.

        Args:
            data: Dict as returned by :meth:`to_dict`.

        Returns:
            New :class:`OpenMMSimulationConfig` instance.
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
    def from_json(cls, path: PathLike) -> "OpenMMSimulationConfig":
        """Load a configuration from a JSON file.

        Args:
            path: Path to a JSON file previously written by :meth:`to_json`.

        Returns:
            New :class:`OpenMMSimulationConfig` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OpenMMEngine(Engine):
    """OpenMM molecular dynamics engine.

    Generates a complete set of OpenMM input files from MolPy
    :class:`~molpy.core.Frame` and :class:`~molpy.core.forcefield.ForceField`
    objects.  OpenMM itself is **not** required for input generation; it is
    only needed for :meth:`serialize_system`.

    The generated Python script can be executed directly::

        python simulate.py

    or via the engine::

        result = engine.run(paths["script"], workdir="./output")

    Example:
        >>> from molpy.engine import OpenMMEngine, OpenMMSimulationConfig
        >>>
        >>> config = OpenMMSimulationConfig(ensemble="NVT", n_steps=10_000)
        >>> engine = OpenMMEngine(check_executable=False)
        >>> paths = engine.generate_inputs(frame, ff, config, "./output")
        >>> # paths["pdb"], paths["forcefield"], paths["script"]
    """

    def __init__(
        self,
        executable: str = "python",
        *,
        workdir: str | Path | None = None,
        launcher: list[str] | None = None,
        env_vars: dict[str, str] | None = None,
        env: str | None = None,
        env_manager: str | None = None,
        check_executable: bool = True,
    ) -> None:
        """Initialise the OpenMM engine.

        Args:
            executable: Python interpreter used when :meth:`run` executes the
                generated simulation script.  Defaults to ``"python"``.
            workdir: Default working directory for :meth:`run`.
            launcher: MPI / scheduler prefix, e.g. ``["mpirun", "-np", "4"]``.
                Prepended before *executable* when running the script.
            env_vars: Extra environment variables forwarded to the subprocess.
            env: Conda / virtual-environment name to activate.
            env_manager: Environment manager (``"conda"`` supported).
            check_executable: Verify *executable* is on PATH at construction.
                Set ``False`` when only using :meth:`generate_inputs`.
        """
        super().__init__(
            executable=executable,
            workdir=workdir,
            launcher=launcher,
            env_vars=env_vars,
            env=env,
            env_manager=env_manager,
            check_executable=check_executable,
        )

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return ``"OpenMM"``.

        Returns:
            Engine identifier string.
        """
        return "OpenMM"

    def _get_default_extension(self) -> str:
        """Return ``".py"`` — the extension for generated simulation scripts.

        Returns:
            ``".py"``
        """
        return ".py"

    def _execute(
        self,
        run_dir: Path,
        capture_output: bool = False,
        check: bool = True,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess:
        """Run the generated Python simulation script.

        Builds the command::

            [launcher...] python simulate.py

        OpenMM must be importable in the configured Python interpreter.

        Args:
            run_dir: Directory containing the simulation files; used as
                ``cwd`` for the subprocess.
            capture_output: Capture stdout/stderr.
            check: Raise :exc:`subprocess.CalledProcessError` on failure.
            timeout: Timeout in seconds.
            **kwargs: Ignored (reserved for future use).

        Returns:
            :class:`subprocess.CompletedProcess`.

        Raises:
            RuntimeError: If no input script has been registered.
            subprocess.CalledProcessError: If *check* is ``True`` and the
                script exits with a non-zero code.
            subprocess.TimeoutExpired: If *timeout* is exceeded.
        """
        if self.input_script is None or self.input_script.path is None:
            raise RuntimeError(
                "No input script found.  Call generate_inputs() or pass a "
                "script to run() first."
            )
        command = self._build_full_command([self.input_script.path.name])
        return subprocess.run(
            command,
            cwd=run_dir,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout,
            env=self._merged_env(),
        )

    # ------------------------------------------------------------------
    # Public input-generation API
    # ------------------------------------------------------------------

    def generate_inputs(
        self,
        frame: "Frame",
        forcefield: "ForceField",
        config: OpenMMSimulationConfig,
        output_dir: PathLike,
        *,
        pdb_filename: str = "system.pdb",
        ff_filename: str = "forcefield.xml",
        script_filename: str = "simulate.py",
    ) -> dict[str, Path]:
        """Generate PDB, XML force field, and Python simulation script.

        Does **not** require OpenMM to be installed.

        Args:
            frame: MolPy :class:`~molpy.core.Frame` with atom positions.
            forcefield: MolPy :class:`~molpy.core.forcefield.ForceField`
                containing interaction parameters.
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

        from molpy.io.data.pdb import PDBWriter
        from molpy.io.forcefield.xml import XMLForceFieldWriter

        PDBWriter(pdb_path).write(frame)
        XMLForceFieldWriter(ff_path).write(forcefield)

        script_text = self._render_simulation_script(
            config, pdb_filename=pdb_filename, ff_filename=ff_filename
        )
        script_path.write_text(script_text, encoding="utf-8")

        return {"pdb": pdb_path, "forcefield": ff_path, "script": script_path}

    def serialize_system(
        self,
        frame: "Frame",
        forcefield: "ForceField",
        config: OpenMMSimulationConfig,
        output_dir: PathLike,
        *,
        pdb_filename: str = "system.pdb",
        ff_filename: str = "forcefield.xml",
        script_filename: str = "simulate.py",
        system_xml_filename: str = "system.xml",
        integrator_xml_filename: str = "integrator.xml",
    ) -> dict[str, Path]:
        """Generate inputs and serialise the OpenMM System + Integrator to XML.

        Calls :meth:`generate_inputs` first, then builds OpenMM objects from
        those files and serialises them with ``openmm.XmlSerializer``.  The
        resulting ``system.xml`` and ``integrator.xml`` files can be loaded
        back without re-parsing the force field.

        Args:
            frame: MolPy :class:`~molpy.core.Frame` with atom positions.
            forcefield: MolPy :class:`~molpy.core.forcefield.ForceField`.
            config: Simulation parameters.
            output_dir: Output directory.
            pdb_filename: PDB coordinate file name.
            ff_filename: XML force field file name.
            script_filename: Python simulation script name.
            system_xml_filename: Serialised ``System`` XML file name.
            integrator_xml_filename: Serialised ``Integrator`` XML file name.

        Returns:
            Dictionary extending :meth:`generate_inputs` output with keys
            ``"system_xml"`` and ``"integrator_xml"``.

        Raises:
            ImportError: If OpenMM is not installed.
        """
        try:
            import openmm.app as app
            import openmm.unit as unit
            from openmm import LangevinMiddleIntegrator, XmlSerializer
        except ImportError as exc:
            raise ImportError(
                "OpenMM is required for serialize_system(). "
                "Install it with: conda install -c conda-forge openmm"
            ) from exc

        out = Path(output_dir)
        paths = self.generate_inputs(
            frame,
            forcefield,
            config,
            output_dir,
            pdb_filename=pdb_filename,
            ff_filename=ff_filename,
            script_filename=script_filename,
        )

        pdb = app.PDBFile(str(paths["pdb"]))
        omm_ff = app.ForceField(str(paths["forcefield"]))
        nonbonded_method = getattr(app, config.nonbonded_method)
        constraints = (
            None if config.constraints == "None" else getattr(app, config.constraints)
        )

        system = omm_ff.createSystem(
            pdb.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=config.nonbonded_cutoff_nm * unit.nanometer,
            constraints=constraints,
        )
        integrator = LangevinMiddleIntegrator(
            config.temperature * unit.kelvin,
            config.friction_per_ps / unit.picosecond,
            config.timestep_fs * unit.femtoseconds,
        )

        system_xml_path = out / system_xml_filename
        integrator_xml_path = out / integrator_xml_filename
        system_xml_path.write_text(XmlSerializer.serialize(system), encoding="utf-8")
        integrator_xml_path.write_text(
            XmlSerializer.serialize(integrator), encoding="utf-8"
        )

        return {
            **paths,
            "system_xml": system_xml_path,
            "integrator_xml": integrator_xml_path,
        }

    # ------------------------------------------------------------------
    # Private script rendering
    # ------------------------------------------------------------------

    def _render_simulation_script(
        self,
        config: OpenMMSimulationConfig,
        *,
        pdb_filename: str,
        ff_filename: str,
    ) -> str:
        """Render a self-contained Python simulation script.

        Dispatches to the ensemble-specific renderer.  Uses f-strings for
        value substitution so that OpenMM unit expressions such as
        ``300*unit.kelvin`` are never misinterpreted as format tokens.

        Args:
            config: Simulation parameters.
            pdb_filename: PDB filename embedded in the script.
            ff_filename: Force field XML filename embedded in the script.

        Returns:
            Complete Python script as a string.
        """
        if config.ensemble == "minimize":
            return self._render_minimize_script(config, pdb_filename, ff_filename)
        if config.ensemble == "NPT":
            return self._render_npt_script(config, pdb_filename, ff_filename)
        return self._render_nvt_script(config, pdb_filename, ff_filename)

    @staticmethod
    def _common_header(config: OpenMMSimulationConfig) -> str:
        """Render the import block common to all ensembles."""
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
            "    LangevinMiddleIntegrator,\n"
            "    Platform,\n"
            "    unit,\n"
            ")\n"
            "import sys\n"
        )

    @staticmethod
    def _system_setup(
        config: OpenMMSimulationConfig, pdb_filename: str, ff_filename: str
    ) -> str:
        """Render PDB loading and system creation lines."""
        constraints_arg = "None" if config.constraints == "None" else config.constraints
        return (
            f'\npdb = PDBFile("{pdb_filename}")\n'
            f'ff  = ForceField("{ff_filename}")\n'
            "system = ff.createSystem(\n"
            "    pdb.topology,\n"
            f"    nonbondedMethod={config.nonbonded_method},\n"
            f"    nonbondedCutoff={config.nonbonded_cutoff_nm}*unit.nanometer,\n"
            f"    constraints={constraints_arg},\n"
            ")\n"
        )

    @staticmethod
    def _integrator_and_simulation(config: OpenMMSimulationConfig) -> str:
        """Render integrator construction and Simulation setup lines."""
        return (
            f"integrator = LangevinMiddleIntegrator(\n"
            f"    {config.temperature}*unit.kelvin,\n"
            f"    {config.friction_per_ps}/unit.picosecond,\n"
            f"    {config.timestep_fs}*unit.femtoseconds,\n"
            ")\n"
            f'platform = Platform.getPlatformByName("{config.platform}")\n'
            "simulation = Simulation(pdb.topology, system, integrator, platform)\n"
            "simulation.context.setPositions(pdb.positions)\n"
        )

    @staticmethod
    def _reporters(config: OpenMMSimulationConfig) -> str:
        """Render reporter registration lines."""
        return (
            f'simulation.reporters.append(DCDReporter("{config.output_dcd}", {config.dcd_reporter_interval}))\n'
            "simulation.reporters.append(StateDataReporter(\n"
            f'    "{config.output_log}", {config.state_reporter_interval},\n'
            "    step=True, potentialEnergy=True, temperature=True, speed=True,\n"
            "))\n"
            f'simulation.reporters.append(CheckpointReporter("{config.output_chk}", {config.checkpoint_interval}))\n'
        )

    def _render_nvt_script(
        self,
        config: OpenMMSimulationConfig,
        pdb_filename: str,
        ff_filename: str,
    ) -> str:
        """Render a Langevin NVT (or NVE) simulation script."""
        parts = [
            self._common_header(config),
            self._system_setup(config, pdb_filename, ff_filename),
            self._integrator_and_simulation(config),
            "\nprint('Minimising energy...')\n",
            "simulation.minimizeEnergy()\n\n",
            self._reporters(config),
            f"\nprint('Running NVT for {config.n_steps} steps...')\n",
            f"simulation.step({config.n_steps})\n",
            "print('Done.')\n",
        ]
        return "".join(parts)

    def _render_npt_script(
        self,
        config: OpenMMSimulationConfig,
        pdb_filename: str,
        ff_filename: str,
    ) -> str:
        """Render an NPT simulation script with a Monte Carlo barostat."""
        barostat_block = (
            "from openmm import MonteCarloBarostat\n"
            "system.addForce(MonteCarloBarostat(\n"
            f"    {config.pressure}*unit.bar,\n"
            f"    {config.temperature}*unit.kelvin,\n"
            f"    {config.barostat_frequency},\n"
            "))\n"
        )
        parts = [
            self._common_header(config),
            self._system_setup(config, pdb_filename, ff_filename),
            barostat_block,
            self._integrator_and_simulation(config),
            "\nprint('Minimising energy...')\n",
            "simulation.minimizeEnergy()\n\n",
            self._reporters(config),
            f"\nprint('Running NPT for {config.n_steps} steps...')\n",
            f"simulation.step({config.n_steps})\n",
            "print('Done.')\n",
        ]
        return "".join(parts)

    def _render_minimize_script(
        self,
        config: OpenMMSimulationConfig,
        pdb_filename: str,
        ff_filename: str,
    ) -> str:
        """Render an energy-minimisation-only script."""
        parts = [
            self._common_header(config),
            self._system_setup(config, pdb_filename, ff_filename),
            self._integrator_and_simulation(config),
            "\nprint('Minimising energy...')\n",
            "simulation.minimizeEnergy(\n",
            f"    tolerance={config.minimize_tolerance}*unit.kilojoule/unit.mole/unit.nanometer,\n",
            f"    maxIterations={config.minimize_max_iterations},\n",
            ")\n",
            "print('Done.')\n",
        ]
        return "".join(parts)
