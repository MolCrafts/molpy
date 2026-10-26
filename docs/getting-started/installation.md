# Installation

MolPy requires Python 3.12 or newer. Install it with pip:

```bash
pip install molcrafts-molpy
```

## Optional dependencies (extras)

MolPy keeps the core lightweight. Install extras when you need integrations or documentation tooling:

| Group | Command | When to use |
|-------|---------|-------------|
| **All** | `pip install molcrafts-molpy[all]` | Everything below |
| **Dev** | `pip install molcrafts-molpy[dev]` | Build, tests, coverage, type checks, lint |
| **Docs** | `pip install molcrafts-molpy[doc]` | Build documentation locally |

## Nightly builds

Nightly builds are paused while releases are frozen; the last snapshots stay on the **separate** PyPI project `molcrafts-molpy-nightly`.
Each build is versioned `X.Y.Z.devN` (a PEP 440 dev release), so you opt in
with `--pre`:

```bash
pip install --pre molcrafts-molpy-nightly
```

The nightly package imports as `molpy`, exactly like the stable one — which means
the two **cannot be installed at the same time** (the same trade-off as
`tensorflow` vs `tf-nightly`). Uninstall `molcrafts-molpy` first, or use a
dedicated virtual environment for nightly testing.

## Native core

The native core ships inside the `molcrafts-molpy` wheel as the extension
module `molpy._core`; there is no separate backend package to install or keep
in step. `mp.__version__` is the version of the core the wheel was built with.


## Verify installation

Run a tiny import check to confirm the package is available and see where it is installed.

```python
import molpy as mp

print("MolPy:", mp.__version__)
print("Released on:", mp.release_date)
```

## Next steps

- Continue with the [Quickstart](quickstart.md) to build, type, and export your first system.
- Then: the [data-model tutorials](../tutorials/index.md), or [Guides](../user-guide/index.md) for task-oriented recipes.
