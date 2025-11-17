# Development Setup

This page describes a typical development environment for working on MolPy.

---

## 1. Clone the repository

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
```

---

## 2. Create and activate a virtual environment

Use your preferred tool (`venv`, `conda`, `poetry`, etc.). Example with `venv`:

```bash
python -m venv .venv
source .venv/bin/activate   # on Linux/macOS
# .venv\Scripts\activate    # on Windows (PowerShell/CMD)
```

---

## 3. Install MolPy in editable mode

```bash
pip install -U pip
pip install -e .[dev]
```

The `[dev]` extra (if defined) should pull in testing and linting tools
used by the project.

---

## 4. Run tests

Make sure the test suite passes locally before making changes:

```bash
pytest
```

You can run specific subsets during development:

```bash
pytest tests/test_core/test_frame.py
pytest -k \"lammps\"        # run tests whose names contain 'lammps'
```

---

## 5. Optional tools

Depending on what you work on, you may also want:

- A recent version of **LAMMPS**, **Packmol**, or other external tools for
  local experiments (not required for unit tests).
- **freud**, **RDKit**, or other third‑party libraries used by analysis and
  adapter modules.

External tools are not strictly required to *build* MolPy, but certain
examples and advanced tests may depend on them.
