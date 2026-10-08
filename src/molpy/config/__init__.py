"""Layered configuration for molpy's engines and wrappers (molcfg).

Every setting an engine (:mod:`molpy.engine`) or a wrapper
(:mod:`molpy.wrapper`) reads from outside its arguments — which executable
to run, the conda / venv environment it lives in, its launcher, its
environment variables and its time limit — resolves here, through four
:mod:`molcfg` layers, later winning:

1. ``defaults`` — molpy's package defaults (:data:`DEFAULTS`);
2. ``user`` — ``~/.molcrafts/molpy/config/config.toml``
   (:func:`user_config_path`; ``MOLCRAFTS_HOME`` moves ``~/.molcrafts``);
3. ``project`` — ``molpy.toml`` in the project directory (the working
   directory unless :func:`load_config` is told otherwise);
4. ``run`` — the overrides passed to :func:`load_config` for one run.

Each value remembers the layer it came from (:meth:`molcfg.Config.meta`), and
:func:`tool_settings` reports it per setting::

    from molpy.config import load_config, tool_settings

    config = load_config({"engine": {"lammps": {"launcher": ["srun"]}}})
    settings = tool_settings(config, "engine.lammps")
    settings.launcher          # ('srun',)
    settings.sources["launcher"]  # ('run', 'engine.lammps.launcher')
"""

from ._config import (
    DEFAULTS,
    LAYERS,
    PROJECT_CONFIG_NAME,
    ToolSettings,
    load_config,
    tool_settings,
    user_config_path,
)

__all__ = [
    "DEFAULTS",
    "LAYERS",
    "PROJECT_CONFIG_NAME",
    "ToolSettings",
    "load_config",
    "tool_settings",
    "user_config_path",
]
