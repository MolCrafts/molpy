# Tool

The external programs molpy runs, and the one core they share.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Tool` | The core every engine and AmberTools program shares: settings → environment → executable → env vars → timeout → logger; `command`, `run_command`, `check_executable` | Subclassing through `mp.engine.Engine` |
| `ToolError` | The one failure: `tool`, `returncode`, `tail` (stderr, stdout or log) | Catching a failed, missing or timed-out program |
| `EnvironmentSpec` | The conda / venv a tool runs in, resolved from its settings | Inspecting or building a command prefix |

## Canonical example

```python
# docs: skip — needs LAMMPS
import molpy as mp

config = mp.config.load_config({"engine": {"env": "md-tools", "env_manager": "conda"}})
engine = mp.engine.LammpsEngine(config=config)
try:
    engine.run(mp.engine.Script("in.lmp", "units real\n"), workdir="calc")
except mp.tool.ToolError as error:
    print(error.tool, error.returncode)  # engine.lammps 1
    print(error.tail)
```

## Key behavior

- Two families of tools are built on `Tool`: the engines of
  [`mp.engine`](engine.md) (group `engine`, settings `[engine.<name>]`) and
  the AmberTools programs behind `mp.ff.typifier`, `mp.ff.charge.Am1BccModel`
  and `mp.builder.AmberPolymerBuilder` (group `ambertools`: antechamber,
  parmchk2, prepgen, tleap; settings `[ambertools.<program>]`)
- A tool's executable, environment (`env` + `env_manager`), `env_vars` and
  `timeout` are its `[<group>.<tool>]` settings in molpy's configuration,
  falling back to `[<group>]` ([`mp.config`](config.md))
- Environment isolation is owned by `EnvironmentSpec`; no auto-detection of
  manager type. `env` and `env_manager` are set together, or both omitted for
  the system `PATH`; managers are `conda` and `venv` (one spelling each)
- Safe to instantiate even if the executable is missing
  (`check_executable()` raises, `is_available()` reports it). The lookup runs
  once per instance; a conda lookup (`conda run … which`) is logged at DEBUG
  and time-limited
- Every failure is a `ToolError` (a `RuntimeError`) carrying `tool`
  (`"engine.lammps"`, `"ambertools.tleap"`), `returncode` and `tail`: a
  missing executable, a program that cannot start, a timeout, a non-zero exit
  (`check=True` is the default) or a step that exits 0 without its output
- Every run is logged to `molpy.<group>.<tool>` as mollog records
  (`process started` / `finished` / `failed` / `timed out`) carrying the
  `command`, `cwd`, `returncode` and `elapsed_s`

## Related

- [API: Engine](engine.md)
- [Guide: AmberTools Integration](../user-guide/13_ambertools_integration.md)
- [Guide: Configure and Log a Run](../user-guide/14_configure_and_log.md)
- [API: Config](config.md)

---

## Full API

::: molpy.tool.Tool

::: molpy.tool.ToolError

::: molpy.tool.EnvironmentSpec
