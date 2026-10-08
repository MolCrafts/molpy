# Config

Layered configuration for the engines and wrappers, built on
[molcfg](https://docs.molcrafts.org/molcfg/).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `load_config` | Merge the four layers (`defaults` → `user` → `project` → `run`), validate, freeze | Getting the config an engine or wrapper runs with |
| `tool_settings` | One tool's resolved settings, each with the layer and key it came from | Inspecting what a tool will run |
| `ToolSettings` | `executable`, `env`, `env_manager`, `conda_executable`, `launcher`, `env_vars`, `timeout`, `sources` | Reading resolved settings |
| `user_config_path` | `~/.molcrafts/molpy/config/config.toml` (`MOLCRAFTS_HOME` moves it) | Finding the user layer's file |
| `PROJECT_CONFIG_NAME` | `molpy.toml`, the project layer's file name | Finding the project layer's file |
| `LAYERS` | `("defaults", "user", "project", "run")` | Naming a layer |
| `DEFAULTS` | The package defaults (the `defaults` layer) | Seeing what an unset key means |

## Canonical example

```python
from molpy.config import load_config, tool_settings

config = load_config({"engine": {"lammps": {"launcher": ["srun", "-n", "4"]}}})
settings = tool_settings(config, "engine.lammps")
print(settings.launcher)  # ('srun', '-n', '4')
print(settings.sources["launcher"])  # ('run', 'engine.lammps.launcher')
print(settings.sources["env"])  # ('defaults', 'engine.env') unless a file sets it
```

## Key behavior

- Tables: `[conda]` (`executable`), `[engine]` and `[engine.<name>]` for
  `lammps`, `gromacs`, `openmm`, `cp2k`; `[wrapper]` and `[wrapper.<name>]`
  for `antechamber`, `tleap`, `parmchk2`, `prepgen`, `sander`
- Keys: `executable`, `env`, `env_manager` (`"conda"` / `"venv"`),
  `env_vars`, `timeout` (seconds) and, for engines, `launcher`; a group table
  holds what its tools share, a tool's own key replaces the group's
- Later layers win key by key; tables such as `env_vars` merge across
  layers, lists such as `launcher` are replaced
- An unknown key or a wrong type is refused when the config loads, naming the
  key and the files read; the loaded config is frozen
- Loading and resolving log to `molpy.config` (DEBUG)

## Related

- [Guide: Configure and Log a Run](../user-guide/14_configure_and_log.md)
- [API: Engine](engine.md) · [API: Wrapper](wrapper.md)

---

## Full API

::: molpy.config.load_config

::: molpy.config.tool_settings

::: molpy.config.ToolSettings

::: molpy.config.user_config_path
