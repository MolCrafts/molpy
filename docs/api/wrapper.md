# Wrapper

Subprocess wrappers for external command-line tools.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Wrapper` | Base: run any CLI executable | Generic external tools |
| `AntechamberWrapper` | AMBER antechamber (type + charge assignment) | GAFF atom typing |
| `Parmchk2Wrapper` | AMBER parmchk2 (missing parameter generation) | Force field completion |
| `TLeapWrapper` | AMBER tleap (topology building) | System assembly |
| `PrepgenWrapper` | AMBER prepgen (residue template generation) | Polymer residues |

## Canonical example

```python
from molpy.wrapper import Wrapper

echo = Wrapper(name="echo", exe="echo")
result = echo.run(args=["hello", "world"])
print(result.stdout) # "hello world\n"
print(result.returncode) # 0
```

## Key behavior

- Environment isolation is owned by `EnvSpec` (`env` + `env_manager`); no auto-detection of manager type
- Both parameters must be set together, or both omitted for the system `PATH`
- Supported managers: `conda`, `venv` (aliases: `pip`, `virtualenv`)
- Safe to instantiate even if executable is missing (failure at `.run()` time)
- All wrappers accept `workdir` for controlling working directory

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)
- [Guide: AmberTools Integration](../user-guide/13_ambertools_integration.md)

---

## Full API

### Environment

::: molpy.wrapper.env

### Base

::: molpy.wrapper.base

### Antechamber

::: molpy.wrapper.antechamber

### Prepgen

::: molpy.wrapper.prepgen

### TLeap

::: molpy.wrapper.tleap
