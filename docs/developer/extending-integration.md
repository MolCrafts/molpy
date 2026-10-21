# Adding a Wrapper or Adapter

This page shows how to integrate external CLI tools (wrappers) and external Python libraries (adapters).

## The distinction

| | Wrapper | Adapter |
|--|---------|---------|
| **Crosses** | Execution boundary (subprocess) | Representation boundary (in-memory) |
| **Concerns** | Executable, environment, return codes, files | Field mapping, conversion fidelity |
| **May** | Run subprocesses, produce files | Build one object model from the other |
| **Must not** | Own workflow logic or chemistry semantics | Execute subprocesses, mutate its input, or cache a copy |


## Adding a Wrapper

Subclass `molpy.wrapper.Wrapper` and name its configuration table in `tool`.
The shared tool core (`molpy.wrapper.Tool`) resolves, in order, the settings
(`[wrapper.<tool>]`, falling back to `[wrapper]`), the conda / venv
environment, the executable, the environment variables, the time limit and
the logger (`molpy.wrapper.<tool>`). Any table under `[wrapper]` is a
tool's, so a third-party wrapper is configured exactly like a built-in one;
give it a default executable by overriding `_default_executable()`, or leave
the executable to the configuration.

```python
import subprocess
from pathlib import Path

from molpy.wrapper import Wrapper


class PackmolWrapper(Wrapper):
    """Wrapper for the packmol CLI (reads [wrapper.packmol])."""

    tool = "packmol"

    def _default_executable(self) -> str:
        return "packmol"

    def run_from_input(self, input_text: str) -> subprocess.CompletedProcess[str]:
        """Run packmol on an input deck given on stdin."""
        return self.run(input_text=input_text)
```

```toml
# molpy.toml
[wrapper.packmol]
executable = "/opt/packmol/bin/packmol"
timeout = 600
```

Higher-level methods (like `run_from_input`) are thin conveniences around
`self.run()`. Workflow logic — which files to require, what to do with them —
belongs in the calling code, not in the wrapper.

### Key points

- `self.run(args, *, input_text=None, capture_output=True, check=True)` runs
  `[env prefix] executable args` in `self.workdir` (created if needed) and
  returns `subprocess.CompletedProcess`
- Every failure is a `molpy.wrapper.ToolError` carrying `tool`
  (`"wrapper.packmol"`), `returncode` and `tail` (the last lines of stderr,
  else stdout): a non-zero exit (unless `check=False`), a program that cannot
  start, a timeout, a missing executable
- `self.check_executable()` returns the resolved path or raises `ToolError`
  naming the settings to fix; `self.is_available()` is the boolean form. Both
  look the executable up once per instance
- `run_step(tool, output, call)` checks the executable, runs one step and
  raises `ToolError` if the file it must write is missing
- Wrappers are safe to instantiate even if the tool is not installed
- An engine is the same recipe on `molpy.engine.Engine`
  ([Simulation Engines](../user-guide/12_engine.md#adding-a-new-engine-takes-a-table-and-two-methods))


## Adding an Adapter

An adapter is a pair of pure functions in `molpy.adapter`, named after the
library: `to_<lib>(atomistic)` and `from_<lib>(obj)`. There is no base class to
subclass — `to_rdkit` / `from_rdkit` (`src/molpy/adapter/_rdkit.py`) is the
reference.

```python
# docs: skip — ASE is not a molpy dependency
import ase

from molpy.core import Atomistic

_XYZ = ("x", "y", "z")


def to_ase(atomistic: Atomistic) -> ase.Atoms:
    """New ``ase.Atoms`` from ``atomistic``; the input is not touched."""
    atoms = list(atomistic.atoms)
    return ase.Atoms(
        numbers=[int(a["atomic_number"]) for a in atoms],
        positions=[[a[k] for k in _XYZ] for a in atoms],
    )


def from_ase(atoms: ase.Atoms) -> Atomistic:
    """New ``Atomistic`` from ``atoms``; the input is not touched."""
    out = Atomistic()
    for atom in atoms:
        out.add_atom(
            atomic_number=int(atom.number),
            **dict(zip(_XYZ, map(float, atom.position))),
        )
    return out
```

### Key points

- **Build, never mutate.** Each function returns a new object. Do not write
  bookkeeping (ids, tags) onto the caller's structure, and do not keep a cached
  copy of either side — a cache goes stale the moment the user edits one.
- **Carry what the other side can hold.** Map the fields the library owns
  natively; carry the rest in whatever per-atom / per-bond metadata it offers
  (RDKit properties, ASE `arrays` / `info`) so it comes back unchanged.
- **Document what is lost.** List, in the module docstring, every field or
  relation that does not survive `from_<lib>(to_<lib>(x))`.
- **Never** run subprocesses inside an adapter — that is a wrapper's job.
- Test round-trip fidelity: `from_<lib>(to_<lib>(x))` should preserve atom
  count, connectivity, coordinates and every carried component, and neither
  function may change its input.


## Handling optional dependencies

The library is an optional extra, so `molpy.adapter` must import without it.
Import the module that needs it inside a guard that treats *only* the library's
own absence as "not installed" — any other `ModuleNotFoundError` is a broken
install and must surface — and make the names raise a clear `ImportError` on use
(`src/molpy/adapter/__init__.py`):

```python
# docs: skip — illustrative; molpy has no ASE adapter
_ASE_MISSING: ModuleNotFoundError | None = None
try:
    from molpy.adapter._ase import from_ase, to_ase
except ModuleNotFoundError as exc:
    if exc.name != "ase":
        raise
    _ASE_MISSING = exc


def __getattr__(name: str) -> object:
    if name in {"to_ase", "from_ase"} and _ASE_MISSING is not None:
        raise ImportError(
            f"molpy.adapter.{name} needs ASE: pip install ase"
        ) from _ASE_MISSING
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

Never bind a missing name to `None`: the failure then surfaces far away, as
`'NoneType' object is not callable`.


## Checklist

- [ ] Wrapper: subclass `Wrapper`, set `tool`, keep `.run()` calls thin
- [ ] Adapter: subclass `Adapter[I, E]`, implement `_do_sync_to_internal/external`
- [ ] Optional deps: guard imports, fail at usage not import
- [ ] Tests: round-trip fidelity for adapters, `ToolError` on failure for wrappers
- [ ] Add tests in `tests/test_wrapper/` or `tests/test_adapter/`
- [ ] Unit-test wrappers/engines with mocks and script literals (never launch the binary)
