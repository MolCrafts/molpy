"""Load molpy's layered configuration and resolve one tool's settings.

The schema, as TOML (every key optional)::

    [conda]
    executable = "conda"          # the conda that `conda run` uses

    [engine]                      # what every engine inherits ...
    env = "md-tools"              # conda env name / prefix, or venv prefix
    env_manager = "conda"         # "conda" or "venv", together with env
    launcher = ["mpirun", "-np", "16"]
    timeout = 3600                # seconds; no limit when absent
    env_vars = { OMP_NUM_THREADS = "1" }

    [engine.lammps]               # ... unless the engine's own table says
    executable = "lmp_mpi"        # otherwise (lammps, gromacs, openmm, cp2k)

    [ambertools]                  # the same keys, launcher aside, for the
    env = "AmberTools25"          # AmberTools programs (antechamber, tleap,
    env_manager = "conda"         # parmchk2, prepgen)

A key in a tool's table replaces the same key of its group's table; the
group's table holds what all of its tools share (one AmberTools environment
for every AmberTools program).

Every other table under ``[engine]`` or ``[ambertools]`` is a tool's: a
third-party :class:`molpy.engine.Engine` subclass reads
``[engine.<its tool>]`` like a built-in one does. The
built-in tools are the tool tables of :data:`DEFAULTS`, the one place their
default executables are stated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_type_hints

import mollog
from molcfg import (
    Config,
    ConfigLoader,
    DictSource,
    Source,
    TomlFileSource,
    ValidationError,
    project_config_dir,
    validate,
)

#: The layers, lowest precedence first. A layer that is absent (no user or
#: project file, no overrides) is skipped.
LAYERS = ("defaults", "user", "project", "run")

#: The project config's file name, looked up in the project directory.
PROJECT_CONFIG_NAME = "molpy.toml"

#: The user config's file name, in ``project_config_dir("molpy")``.
_USER_CONFIG_NAME = "config.toml"

#: molpy's package defaults: the ``defaults`` layer, and the only place a
#: default is stated. The group tables carry every shared key, so a tool
#: inherits them unless a later layer sets the key in the tool's own table; a
#: tool table carries its executable only, and the tool tables are the
#: built-in tools. ``engine.lammps.executable`` is ``None``: the engine takes
#: the first of ``lmp``, ``lmp_serial``, ``lmp_mpi`` found in its environment.
DEFAULTS: dict[str, Any] = {
    "conda": {"executable": "conda"},
    "engine": {
        "env": None,
        "env_manager": None,
        "env_vars": {},
        "launcher": [],
        "timeout": None,
        "lammps": {"executable": None},
        "gromacs": {"executable": "gmx"},
        "openmm": {"executable": "python"},
        "cp2k": {"executable": "cp2k.psmp"},
    },
    "ambertools": {
        "env": None,
        "env_manager": None,
        "env_vars": {},
        "timeout": None,
        "antechamber": {"executable": "antechamber"},
        "tleap": {"executable": "tleap"},
        "parmchk2": {"executable": "parmchk2"},
        "prepgen": {"executable": "prepgen"},
    },
}

_log = "molpy.config"


# -- schema (molcfg.validate) -------------------------------------------------
# Types only: every default lives in DEFAULTS. A tool's keys are optional (it
# inherits its group's); conda.executable is always set by DEFAULTS.


class _Conda:
    executable: str


class _Shared:
    """The keys the ambertools group table and each program's table may hold."""

    env: str | None
    env_manager: Literal["conda", "venv"] | None
    env_vars: dict[str, str] | None
    timeout: int | float | None


class _EngineShared(_Shared):
    """The keys an engine group table and each engine's table may hold."""

    launcher: list[str] | None


class _AmberToolsTool(_Shared):
    executable: str | None


class _EngineTool(_EngineShared):
    executable: str | None


class _Top:
    conda: _Conda | None
    engine: dict | None
    ambertools: dict | None


#: Group → (its shared keys, a tool table's keys).
_GROUPS: dict[str, tuple[type, type]] = {
    "engine": (_EngineShared, _EngineTool),
    "ambertools": (_Shared, _AmberToolsTool),
}


def _shared_keys(group: str) -> tuple[str, ...]:
    return tuple(get_type_hints(_GROUPS[group][0]))


def _validate(data: Mapping[str, Any]) -> None:
    """Validate a merged config: known keys, right types, tool tables as tables.

    A group's own keys are its shared keys; every other key under it names a
    tool, whose table holds the shared keys plus ``executable``.

    Raises:
        molcfg.ValidationError: Every offending key, at once.
    """
    errors: list[str] = []

    def collect(table: Any, schema: type, prefix: str) -> None:
        try:
            validate(table, schema, prefix, allow_extra=False)
        except ValidationError as exc:
            errors.extend(exc.errors)

    collect(dict(data), _Top, "")
    for group, (shared, tool) in _GROUPS.items():
        table = data.get(group)
        if not isinstance(table, Mapping):
            continue
        keys = _shared_keys(group)
        collect({k: v for k, v in table.items() if k in keys}, shared, group)
        for name, value in table.items():
            if name in keys:
                continue
            if isinstance(value, Mapping):
                collect(dict(value), tool, f"{group}.{name}")
            else:
                errors.append(
                    f"{group}.{name}: unexpected field (not a [{group}] key, "
                    "and not a tool's table)"
                )
    if errors:
        raise ValidationError(errors)


# -- loading ------------------------------------------------------------------


def user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """The user layer's file: ``~/.molcrafts/molpy/config/config.toml``.

    The directory is :func:`molcfg.project_config_dir` ``("molpy")``, which
    honours ``MOLCRAFTS_HOME`` and creates the directory; the file itself may
    not exist.

    Args:
        environ: The environment to resolve ``MOLCRAFTS_HOME`` / ``HOME`` in;
            ``None`` is the process environment.
    """
    return project_config_dir("molpy", environ=environ) / _USER_CONFIG_NAME


def load_config(
    overrides: Mapping[str, Any] | None = None,
    *,
    project_dir: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Resolve molpy's configuration through its four layers.

    Later layers win key by key (molcfg's deep merge): ``defaults``, then
    ``user`` (:func:`user_config_path`, when the file exists), then
    ``project`` (``molpy.toml`` in *project_dir*, when it exists), then
    ``run`` (*overrides*). The merged result is validated against the schema
    in this module's docstring and frozen.

    Args:
        overrides: The ``run`` layer: nested tables shaped like the files,
            e.g. ``{"engine": {"lammps": {"launcher": ["srun"]}}}``.
        project_dir: Where ``molpy.toml`` is looked up; the working directory
            when ``None``.
        environ: The environment the user config directory is resolved in;
            ``None`` is the process environment.

    Returns:
        The frozen :class:`molcfg.Config`; ``config.meta(path)["source"]``
        names the layer a value came from.

    Raises:
        molcfg.ValidationError: A layer sets an unknown key or a value of the
            wrong type. The message names the offending keys.
    """
    sources: list[Source] = [DictSource(DEFAULTS, name="defaults")]
    user = user_config_path(environ)
    if user.is_file():
        sources.append(TomlFileSource(user, name="user"))
    project = (
        Path(project_dir) if project_dir is not None else Path.cwd()
    ) / PROJECT_CONFIG_NAME
    if project.is_file():
        sources.append(TomlFileSource(project, name="project"))
    if overrides:
        sources.append(DictSource(dict(overrides), name="run"))

    config = ConfigLoader(sources).load()
    try:
        _validate(config.to_dict())
    except ValidationError as exc:
        files = {"user": user, "project": project}
        layers = ", ".join(
            f"{source.name} ({files[source.name]})"
            if source.name in files
            else source.name
            for source in sources
        )
        raise ValidationError([*exc.errors, f"layers read: {layers}"]) from exc
    config.freeze()
    mollog.get_logger(_log).debug(
        "config loaded",
        layers=[source.name for source in sources],
        user_config=str(user),
        project_config=str(project),
    )
    return config


# -- one tool's settings ------------------------------------------------------


@dataclass(frozen=True)
class ToolSettings:
    """What one engine or AmberTools program runs with, resolved from a config.

    Attributes:
        tool: The tool's table, ``"<group>.<name>"`` (``"engine.lammps"``).
        executable: The program to run (a name looked up in the
            environment, or a path); ``None`` lets the tool choose.
        env: Conda env name / prefix, or venv prefix; ``None`` is the system
            environment.
        env_manager: ``"conda"`` or ``"venv"``, together with *env*.
        conda_executable: The ``conda`` that ``conda run`` invokes.
        launcher: MPI / scheduler prefix before the executable (engines only;
            empty for the AmberTools programs).
        env_vars: Environment variables set for the subprocess.
        timeout: Seconds before the subprocess is killed; ``None`` is no limit.
        sources: Setting name → ``(layer, path)``: the layer
            (:data:`LAYERS`) and the dotted config key the value was read
            from — the tool's own key, or its group's when the tool's table
            leaves the key unset.
    """

    tool: str
    executable: str | None
    env: str | None
    env_manager: Literal["conda", "venv"] | None
    conda_executable: str
    launcher: tuple[str, ...] = ()
    env_vars: Mapping[str, str] = field(default_factory=dict)
    timeout: float | None = None
    sources: Mapping[str, tuple[str, str]] = field(default_factory=dict)


#: The settings resolved per tool, with the group key a tool falls back to.
_INHERITED = ("env", "env_manager", "env_vars", "timeout")


def tool_settings(config: Config, tool: str) -> ToolSettings:
    """Resolve the settings of *tool* (``"engine.lammps"``, ``"ambertools.tleap"``).

    Each setting is the tool's own key when a layer set it, else its group's
    (``engine.<key>`` / ``ambertools.<key>``); :attr:`ToolSettings.sources`
    records which, and from which layer. Any tool name resolves: a tool with
    no table of its own (a third-party engine that nothing
    configures) gets its group's settings and no executable.

    Args:
        config: A config from :func:`load_config`.
        tool: ``"<group>.<name>"`` with group ``engine`` or ``ambertools``; the
            built-in names are the tool tables of :data:`DEFAULTS`.

    Raises:
        KeyError: *tool* is not ``"<group>.<name>"`` with a known group, or
            its name is one of the group's own keys.
        ValueError: A ``timeout`` is not positive.
    """
    group, _, name = tool.partition(".")
    if group not in _GROUPS or not name or "." in name or name in _shared_keys(group):
        raise KeyError(
            f"unknown tool {tool!r}; a tool is '<group>.<name>' with group "
            + " or ".join(repr(g) for g in _GROUPS)
        )

    sources: dict[str, tuple[str, str]] = {}

    def read(key: str, *, inherit: bool) -> Any:
        paths = [f"{tool}.{key}"] + ([f"{group}.{key}"] if inherit else [])
        for path in paths:
            if path in config:
                meta = config.meta(path) or {}
                sources[key] = (str(meta.get("source", "defaults")), path)
                value = config[path]
                return value.to_dict() if isinstance(value, Config) else value
        return None

    keys = _INHERITED + (("launcher",) if group == "engine" else ())
    values = {key: read(key, inherit=True) for key in keys}
    executable = read("executable", inherit=False)
    conda = _read_conda(config, sources)

    timeout = values["timeout"]
    if timeout is not None and timeout <= 0:
        layer, path = sources["timeout"]
        raise ValueError(f"{path} = {timeout!r} (from {layer}) must be positive")

    settings = ToolSettings(
        tool=tool,
        executable=executable,
        env=values["env"],
        env_manager=values["env_manager"],
        conda_executable=conda,
        launcher=tuple(values.get("launcher") or ()),
        env_vars=dict(values["env_vars"] or {}),
        timeout=float(timeout) if timeout is not None else None,
        sources=sources,
    )
    mollog.get_logger(_log).debug(
        "tool settings resolved",
        tool=tool,
        sources={key: f"{layer}:{path}" for key, (layer, path) in sources.items()},
    )
    return settings


def _read_conda(config: Config, sources: dict[str, tuple[str, str]]) -> str:
    """``conda.executable``, recording its layer under ``conda_executable``.

    :data:`DEFAULTS` always sets it, and the schema refuses a non-string.
    """
    path = "conda.executable"
    meta = config.meta(path) or {}
    sources["conda_executable"] = (str(meta.get("source", "defaults")), path)
    return str(config[path])
