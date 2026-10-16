# Wrapper

Subprocess wrappers for external command-line tools.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Tool` | The core every engine and wrapper shares: settings → environment → executable → env vars → timeout → logger; `command`, `run_command`, `check_executable` | Subclassing through `Wrapper` or `Engine` |
| `ToolError` | The one failure: `tool`, `returncode`, `tail` (stderr, stdout or log) | Catching a failed, missing or timed-out tool |
| `Wrapper` | Base: settings from `[wrapper.<tool>]`, logged subprocess runs, `check=True` by default | Writing a wrapper for another program |
| `AntechamberWrapper` | AMBER antechamber (type + charge assignment) | GAFF atom typing |
| `Parmchk2Wrapper` | AMBER parmchk2 (missing parameter generation) | Force field completion |
| `TleapWrapper` | AMBER tleap (topology building) | System assembly |
| `PrepgenWrapper` | AMBER prepgen (residue template generation) | Polymer residues |
| `SanderWrapper` | AMBER sander: `run_from_mdin(mdin_text, prmtop, inpcrd)` | Running a caller's `mdin` on a prmtop/inpcrd |
| `EnvironmentSpec` | The conda / venv a tool runs in, resolved from its settings | Inspecting or building a command prefix |
| `run_step` | `run_step(tool, output, call)`: run one step, require the file it must write; raises `ToolError` with the tool's output | Chaining tools in a pipeline |

## Canonical example

```python
# docs: skip — needs AmberTools
from molpy.config import load_config
from molpy.wrapper import TleapWrapper

config = load_config({"wrapper": {"env": "AmberTools25", "env_manager": "conda"}})
leap = TleapWrapper("leap_work", config=config)
result = leap.run_from_script("source leaprc.gaff2\nquit\n")
print(result.returncode)  # 0
```

## Key behavior

- A wrapper's executable, environment (`env` + `env_manager`), `env_vars` and
  `timeout` are its `[wrapper.<tool>]` settings in molpy's configuration,
  falling back to `[wrapper]` ([`molpy.config`](config.md)); the constructor
  takes the working directory and, optionally, the `config`
- Environment isolation is owned by `EnvironmentSpec`; no auto-detection of
  manager type. `env` and `env_manager` are set together, or both omitted for
  the system `PATH`; managers are `conda` and `venv` (one spelling each)
- Safe to instantiate even if the executable is missing
  (`check_executable()` raises, `is_available()` reports it; `run_step`
  refuses to start). The lookup runs once per instance; a conda lookup
  (`conda run … which`) is logged at DEBUG and time-limited
- Every failure is a `ToolError` (a `RuntimeError`) carrying `tool`
  (`"wrapper.tleap"`), `returncode` and `tail`: a missing executable, a
  program that cannot start, a timeout, a non-zero exit (`check=True` is the
  default; `check=False` returns the `CompletedProcess`) or, in `run_step`, a
  missing output
- A third-party wrapper subclasses `Wrapper`, names its table in `tool` and
  reads `[wrapper.<tool>]` like the built-in ones
  ([Adding a Wrapper](../developer/extending-integration.md))
- Every run is logged to `molpy.wrapper.<tool>` as mollog records
  (`process started` / `finished` / `failed` / `timed out`) carrying the
  `command`, `cwd`, `returncode` and `elapsed_s`

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)
- [Guide: AmberTools Integration](../user-guide/13_ambertools_integration.md)
- [Guide: Configure and Log a Run](../user-guide/14_configure_and_log.md)
- [API: Config](config.md)

---

## Full API

### Environment

::: molpy.wrapper.EnvironmentSpec

### Base

::: molpy.wrapper.Tool

::: molpy.wrapper.ToolError

::: molpy.wrapper.Wrapper

::: molpy.wrapper.run_step

### Antechamber

::: molpy.wrapper.AntechamberWrapper

### Prepgen

::: molpy.wrapper.PrepgenWrapper

::: molpy.wrapper.Parmchk2Wrapper

### TLeap

::: molpy.wrapper.TleapWrapper

### Sander

::: molpy.wrapper.SanderWrapper
