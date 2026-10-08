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

    [wrapper]                     # the same keys, launcher aside, for the
    env = "AmberTools25"          # AmberTools wrappers (antechamber, tleap,
    env_manager = "conda"         # parmchk2, prepgen, sander)

A key in a tool's table replaces the same key of its group's table; the
group's table holds what all of its tools share (one AmberTools environment
for every AmberTools program).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

_ENGINES = ("lammps", "gromacs", "openmm", "cp2k")
_WRAPPERS = ("antechamber", "tleap", "parmchk2", "prepgen", "sander")

#: molpy's package defaults: the ``defaults`` layer. The group tables carry
#: every shared key, so a tool inherits them unless a later layer sets the
#: key in the tool's own table; a tool table carries its executable only.
#: ``engine.lammps.executable`` is ``None``: the engine takes the first of
#: ``lmp``, ``lmp_serial``, ``lmp_mpi`` found in its environment.
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
    "wrapper": {
        "env": None,
        "env_manager": None,
        "env_vars": {},
        "timeout": None,
        "antechamber": {"executable": "antechamber"},
        "tleap": {"executable": "tleap"},
        "parmchk2": {"executable": "parmchk2"},
        "prepgen": {"executable": "prepgen"},
        "sander": {"executable": "sander"},
    },
}

_log = "molpy.config"


# -- schema (molcfg.validate) -------------------------------------------------


class _Conda:
    executable: str = "conda"


class _Shared:
    """The keys a group table and each of its tools' tables may hold."""

    env: str | None = None
    env_manager: Literal["conda", "venv"] | None = None
    env_vars: dict[str, str] = {}
    timeout: int | float | None = None


class _EngineShared(_Shared):
    launcher: list[str] = []


class _Wrapper(_Shared):
    executable: str | None = None


class _Engine(_EngineShared):
    executable: str | None = None


class _Engines(_EngineShared):
    lammps: _Engine | None = None
    gromacs: _Engine | None = None
    openmm: _Engine | None = None
    cp2k: _Engine | None = None


class _Wrappers(_Shared):
    antechamber: _Wrapper | None = None
    tleap: _Wrapper | None = None
    parmchk2: _Wrapper | None = None
    prepgen: _Wrapper | None = None
    sander: _Wrapper | None = None


class _Schema:
    conda: _Conda | None = None
    engine: _Engines | None = None
    wrapper: _Wrappers | None = None


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
        validate(config.to_dict(), _Schema, allow_extra=False)
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
    """What one engine or wrapper runs with, resolved from a config.

    Attributes:
        tool: The tool's table, ``"<group>.<name>"`` (``"engine.lammps"``).
        executable: The program to run (a name looked up in the
            environment, or a path); ``None`` lets the tool choose.
        env: Conda env name / prefix, or venv prefix; ``None`` is the system
            environment.
        env_manager: ``"conda"`` or ``"venv"``, together with *env*.
        conda_executable: The ``conda`` that ``conda run`` invokes.
        launcher: MPI / scheduler prefix before the executable (engines only;
            empty for wrappers).
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
    """Resolve the settings of *tool* (``"engine.lammps"``, ``"wrapper.tleap"``).

    Each setting is the tool's own key when a layer set it, else its group's
    (``engine.<key>`` / ``wrapper.<key>``); :attr:`ToolSettings.sources`
    records which, and from which layer.

    Args:
        config: A config from :func:`load_config`.
        tool: ``"<group>.<name>"`` with group ``engine`` (``lammps``,
            ``gromacs``, ``openmm``, ``cp2k``) or ``wrapper``
            (``antechamber``, ``tleap``, ``parmchk2``, ``prepgen``,
            ``sander``).

    Raises:
        KeyError: *tool* is not a known engine or wrapper.
        ValueError: A ``timeout`` is not positive.
    """
    group, _, name = tool.partition(".")
    known = {"engine": _ENGINES, "wrapper": _WRAPPERS}.get(group, ())
    if name not in known:
        raise KeyError(
            f"unknown tool {tool!r}; known: "
            + ", ".join(f"engine.{n}" for n in _ENGINES)
            + ", "
            + ", ".join(f"wrapper.{n}" for n in _WRAPPERS)
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
    """``conda.executable``, recording its layer under ``conda_executable``."""
    path = "conda.executable"
    meta = config.meta(path) or {}
    sources["conda_executable"] = (str(meta.get("source", "defaults")), path)
    return str(config.get(path, "conda"))
