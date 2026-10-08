# Wrapper

Subprocess wrappers for external command-line tools.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Wrapper` | Base: settings from `[wrapper.<tool>]`, logged subprocess runs | Writing a wrapper for another AmberTools program |
| `AntechamberWrapper` | AMBER antechamber (type + charge assignment) | GAFF atom typing |
| `Parmchk2Wrapper` | AMBER parmchk2 (missing parameter generation) | Force field completion |
| `TleapWrapper` | AMBER tleap (topology building) | System assembly |
| `PrepgenWrapper` | AMBER prepgen (residue template generation) | Polymer residues |
| `SanderWrapper` | AMBER sander (energy minimisation) | Relaxing a prmtop/inpcrd |
| `EnvironmentSpec` | The conda / venv a tool runs in, resolved from its settings | Inspecting or building a command prefix |
| `run_step` | `run_step(tool, output, call)`: run one step, require the file it must write; raises `RuntimeError` with the tool's output | Chaining tools in a pipeline |

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
- Safe to instantiate even if the executable is missing (`check()` /
  `is_available()` report it; `run_step` refuses to start)
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
