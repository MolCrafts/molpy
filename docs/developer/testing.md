# Testing

MolPy uses pytest. Tests live under `tests/`, mirroring the package structure: `tests/test_core/`, `tests/test_io/`, `tests/test_parser/`, and so on.


## Running tests

```bash
pytest tests/ -v                                       # standard local run
pytest tests/test_core/test_frame.py -v                # one file
pytest tests/test_core/test_frame.py::test_creation -v # one test
pytest tests/ -k "lammps" -v                           # keyword filter
pytest --cov=src/molpy tests/ -v --cov-report=term     # with coverage
```

The suite is pure unit tests: wrappers and engines mock `subprocess`; they never
require LAMMPS, Packmol, AmberTools, or RDKit on the machine.


## What to test

Every new behavior needs a test. Cover four categories:

1. **Happy path** — does it produce the right result with normal input?
2. **Edge cases** — empty input, single-element input, boundary values
3. **Error handling** — does it raise the right exception with wrong input?
4. **Regression** — if fixing a bug, add a test that fails without the fix

For MolPy-specific code, two additional patterns are important:

**Immutability checks** — verify that operations return new objects and do not modify the input.

```python
def test_typify_does_not_mutate():
    original = build_test_mol()
    result = typifier.typify(original)
    assert result is not original
    assert len(original.atoms) == original_count
```

**Round-trip tests** — for I/O formats, verify that `write → read → compare` preserves data.

```python
def test_pdb_round_trip(tmp_path):
    write_pdb(tmp_path / "out.pdb", frame)
    restored = read_pdb(tmp_path / "out.pdb")
    assert restored["atoms"].nrows == frame["atoms"].nrows
```


## Wrappers, engines, and docs

- **Wrappers / engines**: mock `subprocess` and assert argv / script *literals*
  written to disk. Never launch antechamber, tleap, lmp, or packmol in unit tests.
- **Doc code blocks**: every ` ```python ` block under `docs/` is executed by
  `tests/test_docs/test_all_doc_blocks.py`. Blocks that shell out or need
  offline artifacts declare `# docs: skip — <reason>` as the first non-empty line.


## Writing good tests

Assert behavior, not implementation. A test should break only when the observable result changes, not when internal code is refactored. Keep fixtures small and focused — a test that sets up 100 atoms to test one bond operation is testing too much. Use the `tmp_path` fixture for file I/O tests to avoid polluting the working directory.
