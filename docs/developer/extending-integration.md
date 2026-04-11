# Adding a Wrapper or Adapter

This page shows how to integrate external CLI tools (wrappers) and external Python libraries (adapters).

## The distinction

| | Wrapper | Adapter |
|--|---------|---------|
| **Crosses** | Execution boundary (subprocess) | Representation boundary (in-memory) |
| **Concerns** | Executable, environment, return codes, files | Field mapping, synchronization fidelity |
| **May** | Run subprocesses, produce files | Hold two object models in sync |
| **Must not** | Own workflow logic or chemistry semantics | Execute subprocesses or produce side effects |


## Adding a Wrapper

Subclass `Wrapper` from `molpy.wrapper.base`. The base class handles executable resolution, conda/virtualenv activation, working directory management, and stdout/stderr capture.

```python
from dataclasses import dataclass, field
from pathlib import Path
from molpy.wrapper.base import Wrapper

@dataclass
class GmxWrapper(Wrapper):
    """Wrapper for GROMACS gmx command."""

    name: str = "gmx"
    exe: str = "gmx"

    def energy_minimize(self, tpr_file: Path) -> Path:
        """Run energy minimization."""
        result = self.run(args=["mdrun", "-s", str(tpr_file), "-deffnm", "em"])
        if result.returncode != 0:
            raise RuntimeError(f"gmx mdrun failed: {result.stderr}")
        return self.workdir / "em.gro"
```

Higher-level methods (like `energy_minimize`) are convenience wrappers around `self.run()`. They should remain thin — workflow logic belongs in the calling code, not in the wrapper.

### Key points

- `self.run(args=[...])` executes the command and returns `subprocess.CompletedProcess`
- `self.resolve_executable()` finds the binary on PATH or in the configured conda env
- `self.is_available()` checks if the tool can be found (safe for conditional imports)
- `workdir` is created automatically; all execution happens there
- Wrappers are safe to instantiate even if the tool is not installed


## Adding an Adapter

Subclass `Adapter[InternalT, ExternalT]` from `molpy.adapter.base`. Implement `_do_sync_to_internal()` and `_do_sync_to_external()`.

```python
from molpy.adapter.base import Adapter
from molpy.core.atomistic import Atomistic

class AseAdapter(Adapter[Atomistic, "ase.Atoms"]):
    """Sync between MolPy Atomistic and ASE Atoms."""

    def _do_sync_to_external(self):
        """Atomistic → ASE Atoms."""
        import ase
        symbols = [a.get("element") for a in self._internal.atoms]
        positions = [[a["x"], a["y"], a["z"]] for a in self._internal.atoms]
        self._external = ase.Atoms(symbols=symbols, positions=positions)

    def _do_sync_to_internal(self):
        """ASE Atoms → Atomistic."""
        mol = Atomistic()
        for atom in self._external:
            mol.def_atom(
                element=atom.symbol,
                x=atom.position[0],
                y=atom.position[1],
                z=atom.position[2],
            )
        self._internal = mol
```

### Key points

- `get_external()` auto-syncs if external is `None` and internal is set (and vice versa)
- Optional imports: if ASE is not installed, the adapter module should fail gracefully at import time
- **Never** run subprocesses inside an adapter — that is a wrapper's job
- Test round-trip fidelity: `internal → external → internal` should preserve atom count, connectivity, and coordinates


## Handling optional dependencies

Follow the existing pattern for optional imports:

```python
# In adapter module
try:
    import ase
    _HAS_ASE = True
except ImportError:
    _HAS_ASE = False
    ase = None

class AseAdapter(Adapter[Atomistic, "ase.Atoms"]):
    def __init__(self, **kwargs):
        if not _HAS_ASE:
            raise ImportError("ASE is required: pip install ase")
        super().__init__(**kwargs)
```

This way the module can be imported without the dependency, and the error only triggers when someone actually tries to use it.


## Checklist

- [ ] Wrapper: subclass `Wrapper`, keep `.run()` calls thin
- [ ] Adapter: subclass `Adapter[I, E]`, implement `_do_sync_to_internal/external`
- [ ] Optional deps: guard imports, fail at usage not import
- [ ] Tests: round-trip fidelity for adapters, return-code checking for wrappers
- [ ] Add tests in `tests/test_wrapper/` or `tests/test_adapter/`
- [ ] Mark tests requiring external tools with `@pytest.mark.external`
