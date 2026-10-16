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

Subclass `molpy.wrapper.Wrapper`. The base class handles executable resolution, conda/virtualenv activation, working directory management, and stdout/stderr capture.

```python
from dataclasses import dataclass, field
from pathlib import Path
from molpy.wrapper import Wrapper

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
        symbols=[a["element"] for a in atoms],
        positions=[[a[k] for k in _XYZ] for a in atoms],
    )


def from_ase(atoms: ase.Atoms) -> Atomistic:
    """New ``Atomistic`` from ``atoms``; the input is not touched."""
    out = Atomistic()
    for atom in atoms:
        out.def_atom(element=atom.symbol, **dict(zip(_XYZ, map(float, atom.position))))
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

- [ ] Wrapper: subclass `Wrapper`, keep `.run()` calls thin
- [ ] Adapter: a `to_<lib>` / `from_<lib>` pair of pure functions; document what does not round-trip
- [ ] Optional deps: guard imports on `exc.name`, fail at usage not import
- [ ] Tests: round-trip fidelity for adapters, return-code checking for wrappers
- [ ] Add tests in `tests/test_wrapper/` or `tests/test_adapter/`
- [ ] Unit-test wrappers/engines with mocks and script literals (never launch the binary)
