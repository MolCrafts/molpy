[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/getting-started/installation.ipynb)

# Installation

MolPy is a pure Python package. Install it with pip:

```bash
pip install molcrafts-molpy
```

## Optional dependencies (extras)

MolPy keeps the core lightweight. Install extras when you need integrations or documentation tooling:

| Group | Command | When to use |
|-------|---------|-------------|
| **All** | `pip install molcrafts-molpy[all]` | Everything below |
| **Dev** | `pip install molcrafts-molpy[dev]` | Tests, coverage, tooling |
| **Docs** | `pip install molcrafts-molpy[doc]` | Build documentation locally |

## Nightly builds

Bleeding-edge snapshots are published to a **separate** PyPI project,
`molcrafts-molpy-nightly`, on every push to the `nightly` branch. Each build is
versioned `X.Y.Z.devN` (a PEP 440 dev release), so you opt in with `--pre`:

```bash
pip install --pre molcrafts-molpy-nightly
```

The nightly package imports as `molpy`, exactly like the stable one — which means
the two **cannot be installed at the same time** (the same trade-off as
`tensorflow` vs `tf-nightly`). Uninstall `molcrafts-molpy` first, or use a
dedicated virtual environment for nightly testing.

## Verify installation

Run a tiny import check to confirm the package is available and see where it is installed.

```python
import molpy as mp

print("MolPy:", mp.version)
print("Released on:", mp.release_date)
```

## Next steps

- Continue with the Quickstart to build, type, and export your first system.
- Read Core Concepts to understand why MolPy's data model is split into graph (`Atomistic`) vs arrays (`Frame`).
- Continue with Concepts for the object model, or jump into Guides for larger, task-oriented workflows.
