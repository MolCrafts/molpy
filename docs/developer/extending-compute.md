# Adding a Compute Operation

This page shows how a new analysis reaches MolPy users. An *analysis
operation* — a **compute** — is anything that turns simulation data into a
number, a curve, or a grid: a radial distribution function from neighbour
tables, a mean squared displacement from a trajectory, a dipole series from
positions and charges.

!!! important "Science lives in molrs"
    Every shipped kernel is implemented once in **molrs** and re-exported by
    identity into `molpy.compute`. molpy holds no compute code: a new analysis
    for every user is a new type in `molrs::compute`, with its Python binding,
    and it appears on `molpy.compute` with no molpy change at all. Do **not**
    add a Python recipe class that reimplements Green–Kubo, Einstein
    conductivity, or a dielectric spectrum, and do not wrap a molrs class in a
    molpy one.

    A time correlation in particular is never a new kernel. molrs has one
    estimator, `Correlator`, and every correlation — `Msd`, `Vacf`,
    `ReorientationAcf`, `Survival`, the dipole and current ACFs — is a
    configuration of it that fixes the settings defining the quantity and
    returns a `TimeCorrelation`. A new quantity of that shape is either a
    `Correlator` built directly in your script or, if chemists know it by
    name, one more such configuration in molrs.

## The contract: configuration in, data through `compute`

There is **no base class to subclass**. A compute is an ordinary class obeying
two rules:

1. **Configuration goes to `__init__`.** These are the values that define the
   measurement — `n_bins`, `r_max`, `max_lag`, a window, a tolerance. Nothing
   that is being analysed belongs there: no positions, no charges, no atom
   index lists, no reference structure.
2. **Data goes to `compute`, by keyword.** Arrays, the sample interval `dt`,
   a `Box`, neighbour tables, a partition, the result of another compute. Each
   input is a named keyword argument with a concrete type, so a call site
   reads `Msd(max_lag=500).compute(positions=xyz, dt=10.0)`.

That split is the whole point: one configured object can be applied to many
trajectories with the measurement held fixed, and the parameters that decide
what the number *means* are visible at construction rather than buried in a
call.

A compute never reads a `Frame` and never names a column. Which columns feed
an analysis — unwrapped or wrapped positions, which charges, which molecules —
is the caller's decision, made with the Frame/Block API before the call. A
compute that looked up `"x"` or `"q"` itself would hard-code one schema and
fail on every other.

`molpy.compute.Compute` is the name of that contract, and it is **not** a base
class. It is a [`typing.Protocol`](https://peps.python.org/pep-0544/) — a
*structural* type — owned by the molrs backend and re-exported unchanged.
Structural means a class conforms by **having the right method**, never by
inheriting one: define `compute(...)` and your class already is a `Compute`,
with no import, no registration, and no subclassing.

The protocol is decorated `@runtime_checkable`, so `isinstance(obj, Compute)`
runs — but read what it asserts. A runtime protocol check tests for the
*presence* of the method and nothing else: not its signature, not its
parameter types, not its behaviour. Treat it as a smoke test at a boundary,
not as dispatch in a loop.

## A shipped analysis is added in molrs

An analysis every user should have goes into `molrs::compute` as one type
implementing the core `Compute` trait, with its configuration in `new` /
`with_*`, its data in a per-type input struct, its defaults in the core, and a
`ComputeSpec` descriptor. The Python binding is derived from that type, so the
class lands on `molrs.compute` and therefore on `molpy.compute` under the same
name. molpy's test suite checks the mirror by identity over the whole of
`molrs.compute.__all__`; no molpy file names the new class.

Results chain without adapters. A fit reads the `TimeCorrelation` a
correlation returns, `Pca` reads the `SteinhardtResult` a descriptor returns,
and `CenterOfMass` reads a `ClusterResult` as its partition. A new type should
take and give these same objects rather than a molpy-side dictionary.

## A one-off analysis is a class in your script

Some analyses belong to one study, not to the library. Those are written in
the script that uses them, following the same contract, so they compose with
the shipped ones and read the same way.

Take the number-density profile of a slab along one box axis: the count of
atoms in each slice divided by the slice volume. Which axis and how many
slices define the measurement, so they are configuration. The positions and
the box are what is being measured, so they are data:

```python
import numpy as np
from numpy.typing import NDArray

import molpy as mp


class SlabDensity:
    """Number density in slices along one axis of an orthogonal box."""

    def __init__(self, axis: int, n_bins: int) -> None:
        self.axis = axis
        self.n_bins = n_bins

    def compute(self, *, positions: NDArray, simbox: mp.Box) -> NDArray:
        """Atoms per Å³ in each slice.

        Args:
            positions: shape (n_atoms, 3), Å, inside the box.
            simbox: an orthogonal periodic box.

        Returns:
            Density per slice, shape (n_bins,), Å⁻³.
        """
        lengths = simbox.lengths
        edges = np.linspace(0.0, lengths[self.axis], self.n_bins + 1)
        counts, _ = np.histogram(positions[:, self.axis], bins=edges)
        slice_volume = np.prod(lengths) / self.n_bins
        return counts / slice_volume
```

The data arrives by keyword (`*` makes every data argument keyword-only), and
the class reads no frame. Pulling the coordinates out of a frame is the
caller's line:

```python
frame = mp.Frame()
frame["atoms"] = {
    "x": [1.0, 2.0, 3.0, 4.0],
    "y": [1.0, 1.0, 1.0, 1.0],
    "z": [1.0, 1.0, 6.0, 6.0],
}
frame.box = mp.Box(mp.core.Cell.lengths([10.0] * 3))

profile = SlabDensity(axis=2, n_bins=2)
print(profile.compute(positions=frame.coords, simbox=frame.box))  # -> [0.004 0.004]
```

Two atoms in each half of a 1000 Å³ box give $2/500 = 0.004$ Å⁻³ per slice.

Nothing is mutated. The histogram and the division build new arrays rather
than writing into the caller's; a compute that wrote into its input would
silently corrupt every other analysis reading the same frame.

The class was never told about `Compute`, yet it satisfies it:

```python
from molpy.compute import Compute

print(isinstance(profile, Compute))             # -> True
```

## Design rules

1. **Configuration goes to `__init__`** — stored as plain attributes. No
   `**config` catch-all: a misspelled keyword must raise `TypeError` at the
   call site, never disappear into a dictionary. No data either.
2. **Data goes to `compute`, by keyword** — one typed keyword per input, so
   the signature documents what the analysis consumes.
3. **No frames, no field names** — the caller reads columns through the
   Frame/Block API and hands arrays over.
4. **No mutation** — `compute` returns new objects and never writes into the
   arrays it was handed.
5. **One verb** — a fit, a check and a transform are computes too; there is no
   second method that computes the same thing, and no free function.
6. **Keep `compute` focused** — one named measurement, not a workflow. Turning
   a raw curve into a transport coefficient (fit window, SI prefactor) is the
   caller's composition step and belongs in a script where a reviewer can see
   it.
7. **No silent fallback** — out-of-range input is an error, never a clamped
   or invented value.
8. **State units** for every physical argument and every returned quantity.
   Analysis units here are LAMMPS *real*: Å, e, fs, K.

## Checklist

- [ ] A shipped analysis is a `molrs::compute` type with a `ComputeSpec`; molpy
      gains it by identity and needs no code
- [ ] Plain class: no base class, no `super().__init__`, no `dump()`
- [ ] Configuration only in `__init__`, stored as attributes
- [ ] `def compute(self, *, …)` with type hints — one keyword per data input
- [ ] No `Frame` argument and no column-name literal
- [ ] Units documented for every physical parameter and return value
- [ ] `isinstance(obj, molpy.compute.Compute)` is `True`
