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
print(result.stdout)    # "hello world\n"
print(result.returncode) # 0
```

## Key behavior

- Wrappers handle conda/virtualenv activation via `env` and `env_manager` parameters
- Safe to instantiate even if executable is missing (failure at `.run()` time)
- All wrappers accept `workdir` for controlling working directory

## Related

- [Concepts: Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md)
- [Guide: AmberTools Integration](../user-guide/07_ambertools_integration.md)

---

## Full API

### Base

::: molpy.wrapper.base

### Antechamber

::: molpy.wrapper.antechamber

### Prepgen

::: molpy.wrapper.prepgen

### TLeap

::: molpy.wrapper.tleap
