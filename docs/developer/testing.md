# Testing

MolPy uses **pytest** for all automated tests. Good tests are critical for
refactoring the core data model and IO layers safely.

This page outlines basic testing expectations.

---

## Where tests live

- All tests go in the top‑level `tests/` directory.
- Mirror the package structure where it makes sense:
  - `tests/test_core/test_frame.py`
  - `tests/test_io/test_pdb.py`
  - `tests/test_reacter/test_basic.py`

This makes it easy to find tests corresponding to a module.

---

## What to test

- **Core types** (`Frame`, `Block`, `Box`, `Atomistic`):
  - Indexing and slicing behavior
  - Serialization (`to_dict`/`from_dict`, IO round‑trips)
  - Edge cases (empty blocks, unusual shapes, etc.)
- **IO**:
  - Round‑trip tests (write -> read -> compare)
  - Handling of minimal and slightly “messy” inputs
- **Domain logic** (builders, reacters, typifiers, packers):
  - Correctness on small, realistic systems
  - That invariants are preserved (e.g. charge conservation, connectivity)

Whenever you fix a bug, add a regression test.

---

## Test data

Prefer:

- Small, realistic molecules (water, methane, short polymers).
- Lightweight example files checked into `tests/data/` when needed.

Avoid:

- Huge input files.
- Tests that depend on network access.

External tools (e.g. Packmol, LAMMPS) should be mocked or guarded so that
tests remain fast and reliable on CI machines that may not have them installed.

---

## Running tests

Basic usage:

```bash
pytest
```

To run a subset:

```bash
pytest tests/test_core/test_frame.py
pytest -k \"rdf\"   # tests whose names contain 'rdf'
```

Keep the test suite fast enough that contributors can run it locally
before opening a PR.
