# Adding a Compute Operation

This page shows how to add a reusable analysis operation to MolPy. An *analysis
operation* — a **compute** — is anything that turns simulation data into a
number, a curve, or a grid: a radial distribution function from a frame and its
neighbour list, a mean squared displacement from a trajectory, a dipole moment
from positions and charges.

!!! important "Science lives in molrs"
    Transport, dielectric, `Vacf`, and spectral kernels are implemented once in
    **molrs** and re-exported (identity) into `molpy.compute`. Do **not** add a
    parallel Python recipe class that reimplements Green–Kubo, Einstein
    conductivity, or dielectric spectra. Prefer a molrs `Compute` + `LinearFit`
    composition; molpy only wraps frame extraction when needed.

## The contract: one method, called `compute`

There is **no base class to subclass**. A compute in molpy is an ordinary
Python class obeying two rules:

1. **Construction parameters go to `__init__`.** These are the numbers that
   define the measurement — `n_bins`, `r_max`, `cutoff`, `use_masses`. Store
   them as plain attributes; nothing else happens at construction.
2. **Data goes to `compute`.** One positional parameter per data input, with a
   concrete, fully typed signature. `compute(frames, neighbors)` takes two
   inputs; there is no single-input restriction and no `*args` bag.

That split is the whole point: one configured object can be applied to many
trajectories with the measurement held fixed, and the parameters that decide
what the number *means* are visible at construction rather than buried in a
call.

`molpy.compute.Compute` is the name of that contract, and it is **not** a base
class. It is a [`typing.Protocol`](https://peps.python.org/pep-0544/) — a
*structural* type — owned by the molrs backend and re-exported unchanged, so
`molpy.compute.Compute is molrs.compute.Compute`. Structural means a class
conforms by **having the right method**, never by inheriting one: define
`compute(...)` and your class already is a `Compute`, with no import, no
registration, and no subclassing. Inheriting from the protocol explicitly would
turn it back into an abstract base class, which is exactly the duplication the
re-export exists to prevent.

The protocol is decorated `@runtime_checkable`, so `isinstance(obj, Compute)`
runs — but read what it asserts. A runtime protocol check tests for the
*presence* of the method and nothing else: not its signature, not its parameter
types, not its behaviour. Treat it as a smoke test at a boundary, not as
dispatch in a loop.

!!! warning "Removed: the old `Compute` base class"
    Earlier versions of molpy shipped their own `Compute` abstract base class
    whose data entry point was `__call__`, with construction parameters
    forwarded to `super().__init__(**config)` and read back by a `dump()`
    method. The base class, the `**config` catch-all, and `dump()` are all
    gone. Name the method `compute`, store your own attributes, and do not call
    `super().__init__`.

## Which shape to use

| Need | Shape | Example |
|------|-------|---------|
| Analysis with a molrs kernel | re-export the molrs class by identity | `Msd`, `Rdf`, `EinsteinConductivity`, `OnsagerCorrelation` |
| Analysis molpy owns (no kernel in molrs) | plain class with `compute(...)` | the `CollectiveDipole` example below |
| Pure array math with no owner | module-level function | `signal.acf_fft` |

A molrs class that already carries its own verb keeps it — `RadicalVoronoi`
builds with `build(...)`, `VoronoiIntegration` with `integrate(...)`,
`LinearFit` with `fit(...)`, `KramersKronig` with `check(...)`. Those are molrs
contracts, not molpy's to rename. Wrapping one of them in a molpy class whose
only content is a one-line forward is a façade, not an operator: re-export the
molrs class instead.

## Writing one

Take configuration in `__init__`, implement `compute` with a concrete typed
signature, and state the unit of every physical quantity in the docstring.

The example below is the collective dipole moment $\mathbf{M} = \sum_i q_i
\mathbf{r}_i$ — the sum of each atom's charge times its position. It is the
input the dielectric and Einstein-conductivity kernels expect, and molpy does
not ship it, because which atoms and which charges belong in the sum is a
question only you can answer.

```python
import numpy as np
from numpy.typing import NDArray


class CollectiveDipole:
    """Collective dipole moment of one configuration."""

    def __init__(self, subtract_center: bool = False) -> None:
        self.subtract_center = subtract_center

    def compute(self, positions: NDArray, charges: NDArray) -> NDArray:
        """Sum q_i r_i over the atoms of one configuration.

        Args:
            positions: shape (n_atoms, 3), Å.
            charges: shape (n_atoms,), elementary charge e.

        Returns:
            Dipole vector, shape (3,), in e·Å.
        """
        if self.subtract_center:
            positions = positions - positions.mean(axis=0)
        return (charges[:, None] * positions).sum(axis=0)
```

Two data inputs, two positional parameters, and the only configuration —
whether positions are referred to the geometric centre before summing — is
fixed once, at construction. It has to be configuration rather than an
afterthought: when the total charge $\sum_i q_i$ is not zero the dipole depends
on where you put the origin, so the choice of origin is part of the
measurement's definition and belongs next to it.

Nothing is mutated. `positions - positions.mean(axis=0)` builds a new array
rather than shifting the caller's; a compute that wrote into its input would
silently corrupt every other analysis reading the same frame.

Usage is the two beats the contract promises — configure, then `.compute(...)`:

```python
positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
charges = np.array([-1.0, 1.0])

dipole = CollectiveDipole()
print(dipole.compute(positions, charges))      # -> [1. 0. 0.]
```

A charge $-e$ at the origin and $+e$ one ångström along $x$ give
$\mathbf{M} = 1\ e\,\text{Å}$ pointing from the negative charge to the positive
one, which is the sign convention the formula above states and the code follows.

The class was never told about `Compute`, yet it satisfies it:

```python
from molpy.compute import Compute

print(isinstance(dipole, Compute))             # -> True
```

## Design rules

1. **Configuration goes to `__init__`** — stored as plain attributes. No
   `**config` catch-all: a misspelled keyword must raise `TypeError` at the
   call site, never disappear into a dictionary.
2. **Runtime data goes to `compute`** — one typed positional per input, so the
   signature documents what the analysis consumes.
3. **No mutation** — `compute` returns new objects and never writes into the
   arrays or the `Frame` it was handed.
4. **Keep `compute` focused** — one named measurement, not a workflow. Turning
   a raw curve into a transport coefficient (fit window, SI prefactor) is the
   caller's composition step and belongs in a script where a reviewer can see
   it.
5. **Do not rename a molrs verb**, and do not wrap a molrs class that needs no
   addition; re-export it.
6. **State units** for every physical argument and every returned quantity.
   Analysis units here are LAMMPS *real*: Å, e, fs, K.
7. **Test in isolation** — hand-written inputs, one behaviour per test.

## Checklist

- [ ] Plain class: no base class, no `super().__init__`, no `dump()`
- [ ] Construction parameters stored as attributes in `__init__`
- [ ] `def compute(self, …)` with type hints — one positional per data input
- [ ] Units documented for every physical parameter and return value
- [ ] `isinstance(obj, molpy.compute.Compute)` is `True`
- [ ] Unit test under `tests/test_compute/`, mirroring the module path
