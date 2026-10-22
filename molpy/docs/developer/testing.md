# Testing

MolPy uses pytest. Tests live under `tests/`, mirroring the package structure
(`src/molpy/io/data/lammps.py` → `tests/test_io/test_data/test_lammps.py`) with no
`__init__.py` files (`--import-mode=importlib`). Fixture files are small,
committed under `tests/tests-data/<format>/`, and reached through the
`TEST_DATA_DIR` fixture; nothing is downloaded.


## Running tests

```bash
uv run --extra dev python -m pytest tests/ -n auto                         # the CI command
uv run --extra dev python -m pytest tests/test_io/test_data/test_lammps.py # one file
uv run --extra dev python -m pytest tests/ -k "lammps"                     # keyword filter
```

The suite is pure unit tests: the engines and the AmberTools programs mock
`subprocess`; nothing requires LAMMPS, GROMACS, OpenMM or AmberTools on the
machine (packing is molpack, pure wheel), and nothing is skipped for their
absence: a test that can only run where a program is installed runs nowhere
CI looks, so it is not written.


## What to test

Every new behavior needs a test. Cover four categories:

1. **Happy path** — does it produce the right result with normal input?
2. **Edge cases** — empty input, single-element input, boundary values
3. **Error handling** — does it raise the right exception with wrong input?
4. **Bug fix** — a unit test of the corrected behaviour that fails without the fix

A test checks correctness: what code computes, writes, raises or refuses.
What is *not* a unit test and does not belong in `tests/`: an end-to-end
pipeline, a doc snippet run as a test, a number captured from another
program, a check that molpy returns what molrs returns, a grep or AST scan
over source text, a check of code style, naming or where a module lives, a
check that an API is absent or has a particular signature, a large corpus
file.

## Reach the public API the way the manual does

A test of the public API writes what a user writes: `import molpy as mp`,
then `mp.<subsystem>.<Name>` (`mp.engine.LammpsEngine`, `mp.io.read_pdb`,
`mp.Frame`). A broken export then fails the test that uses it. An internal
helper is imported from its implementation module
(`from molpy.engine.lammps import _splice_coords`). `tests/test_init.py`
checks that every name an `__all__` lists can be taken.

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
    mp.io.write_pdb(tmp_path / "out.pdb", frame)
    restored = mp.io.read_pdb(tmp_path / "out.pdb")
    assert restored["atoms"].n_rows == frame["atoms"].n_rows
```


## External programs, engines, and docs

- **Engines / AmberTools programs**: mock `subprocess` and assert argv /
  script *literals* written to disk. Never launch antechamber, tleap, or lmp
  in unit tests.
- **Doc code blocks** are not executed by the test gate. Blocks that shell out
  or need offline artifacts still declare `# docs: skip — <reason>` as the first
  non-empty line so a reader knows they are not runnable as-is.


## Writing good tests

Assert behavior, not implementation. A test should break only when the observable result changes, not when internal code is refactored. Keep fixtures small and focused — a test that sets up 100 atoms to test one bond operation is testing too much. Use the `tmp_path` fixture for file I/O tests to avoid polluting the working directory.
