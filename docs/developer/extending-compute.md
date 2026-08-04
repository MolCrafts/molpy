# Adding a Compute Operation

This page shows how to add reusable analysis operations (`Compute`) to MolPy.

!!! important "Science lives in molrs"
    Transport, dielectric, VACF, and spectral kernels are implemented once in
    **molrs** and re-exported (identity) into `molpy.compute`. Do **not** add a
    parallel Python recipe class that reimplements Green–Kubo, Einstein
    conductivity, or dielectric spectra. Prefer a molrs `Compute` + `Fit`
    composition; molpy only wraps frame extraction when needed.

## Which base class to use

| Need | Base class | Example |
|------|-----------|---------|
| Frame-oriented analysis (molrs kernel behind a shell) | `Compute` | `MSD`, `RDF` |
| Array-oriented transport / dielectric | re-export molrs type | `EinsteinConductivity`, `Onsager` |
| Pure array math with no owner | module function | `signal.acf_fft` |

`Compute` is a configurable callable. Construction parameters go to `__init__` and are handed to `super().__init__(**config)` so `dump()` can round-trip them; data inputs go to `__call__`, which is the one abstract method.

## Adding a Compute operation

Subclass `Compute`, take configuration in `__init__`, implement `__call__` with a concrete typed signature — one parameter per data input.

```python
import numpy as np
from numpy.typing import NDArray
from molpy.compute import Compute


class RadiusOfGyration(Compute):
    """Compute radius of gyration for each frame."""

    def __init__(self, use_masses: bool = True) -> None:
        super().__init__(use_masses=use_masses)
        self.use_masses = use_masses

    def __call__(self, positions: NDArray, masses: NDArray | None = None) -> float:
        """Compute Rg for a set of positions.

        Args:
            positions: shape (n_atoms, 3)
            masses: shape (n_atoms,), optional

        Returns:
            Scalar Rg value.
        """
        if self.use_masses and masses is not None:
            w = masses / masses.sum()
        else:
            w = np.ones(len(positions)) / len(positions)

        com = (positions * w[:, None]).sum(axis=0)
        dr = positions - com
        rg2 = (w * (dr**2).sum(axis=1)).sum()
        return float(np.sqrt(rg2))
```

Usage:

```python
rng = np.random.default_rng(0)
positions = rng.uniform(0.0, 10.0, size=(50, 3))
masses = np.full(len(positions), 12.011)

rg = RadiusOfGyration(use_masses=True)
value = rg(positions, masses)
assert rg.dump() == {"use_masses": True}
```

## Design rules

1. **Configuration goes to `__init__`** — set once, and forwarded to `super().__init__` so `dump()` can serialize it
2. **Runtime data goes through `__call__`** — different inputs, same protocol
3. **No mutation** — `__call__` returns new objects, never modifies inputs
4. **Keep `__call__` focused** — one clear task, not a workflow engine
5. **Test in isolation** — each Compute should be testable with synthetic data

## Checklist

- [ ] Subclass `Compute`
- [ ] Pass construction parameters to `super().__init__(**config)`
- [ ] Implement `__call__` with type hints
- [ ] Write tests in `tests/test_compute/`
