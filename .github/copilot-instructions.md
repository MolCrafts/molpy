# Copilot instructions (MolPy)

## Project snapshot (fill later)

- (Intentionally left minimal; expand as needed.)

## Repo layout (what to touch)

- Source: `src/molpy/` (package uses `src/` layout)
- Tests: `tests/` (pytest; external-dependent tests are marked `external`)
- Docs: `docs/` + `zensical.toml` (Zensical; notebooks pre-rendered to Markdown via `scripts/render_notebooks.py`)
- CI/release: `.github/workflows/`

## Local workflows (commands used in CI)

- Install (dev): `pip install -e ".[dev]"`
- Format check: `black --check src/ tests/`
- Tests (unit/default): `pytest tests/ -v -m "not external"`
- Release is tag-driven (`v*`) and validates `src/molpy/version.py` matches the tag.

## Project-specific patterns

- Public API is re-exported from `src/molpy/__init__.py` (keep it import-safe; avoid importing optional deps unguarded).
- Optional integrations should be import-guarded (pattern: `src/molpy/adapter/__init__.py` uses `try/except ModuleNotFoundError`).
- External tests are classified automatically in `tests/conftest.py` (engine suite + RDKit-related tests).
- Some tests may use `TEST_DATA_DIR` fixture which clones/pulls `https://github.com/molcrafts/tests-data.git` on demand; avoid requiring network for non-`external` tests.

## Hard restrictions (must follow)

### Code

* **MUST** add full type hints for all public functions/methods, class attributes, and return types.
* **MUST** keep public APIs type-stable: do not change argument names/types or return types without updating tests and docs in the same patch.
* **MUST** write Google-style docstrings for all public modules, classes, functions, and methods.
* **MUST** use OOP-style design for core behaviors: prefer classes with explicit responsibilities over ad-hoc procedural code.
* **MUST NOT** introduce “factory functions” as the primary user-facing API when a class is the domain concept (allowed only for tiny internal helpers).
* **MUST** keep side effects explicit: functions/methods that mutate must be clearly named and documented; pure transforms must not mutate inputs.
* **MUST** keep transformations deterministic and reproducible: avoid randomness and time-dependent behavior unless explicitly required and seeded/configurable.
* **MUST** keep data models explicit and serializable where applicable (e.g., configs, IR, templates): avoid hidden globals or implicit state.
* **MUST NOT** introduce new dependencies unless strictly necessary; if added, **MUST** justify via a short comment in code and add minimal tests around the integration point.
* **MUST** keep modules small and focused: one file should represent one coherent concept; avoid “utils.py dumping ground”.
* **MUST** maintain backward compatibility for exported symbols unless the change is explicitly requested; if breaking, **MUST** add a migration note in docs.
MUST NOT use try/except for normal control flow or error masking.
try/except is allowed only at explicit external boundaries (e.g., file I/O, subprocess calls, network access, third-party library interaction), and exceptions must be either re-raised or converted into well-defined domain errors.
* **MUST NOT** use `try`/`except` for normal control flow or error masking.
  `try`/`except` is allowed **only** at explicit external boundaries (e.g., file I/O, subprocess calls, network access, third-party library interaction), and exceptions must be either re-raised or converted into well-defined domain errors.
* **MUST NOT** introduce hard-coded logic, special cases, or conditional branches solely to satisfy tests or silence failures during debugging. If a test fails, the underlying logic must be corrected; modifying production code to “fit the test” without a sound semantic reason is strictly forbidden.

## Tests

* **MUST** add tests for every new public class/function and every behavior change.
* **MUST** ensure each public class has at least one dedicated test file or clearly scoped test section (your “one class ↔ one test” rule).
* **MUST** keep tests minimal and non-overlapping: no duplicated assertions for the same behavior.
* **MUST** keep tests complete for the intended behavior: cover success path + at least one failure/edge case for each public API.
* **MUST** write tests that assert observable behavior, not implementation details (no testing private helpers unless unavoidable).
* **MUST** avoid flaky tests: no reliance on network, wall-clock time, random seeds (unless fixed), or external executables (unless explicitly marked as integration tests).
* **MUST** separate unit vs integration tests:

  * Unit tests: pure Python, fast, deterministic.
  * Integration tests: allowed to touch filesystem/subprocess, but **MUST** be clearly marked and minimal.
* **MUST** use meaningful assertions (no “it runs” tests). Every test must assert state, outputs, exceptions, or artifacts.
* **MUST** include negative tests for validation: if inputs are invalid, the API must raise the documented exception type/message pattern.
* **MUST** update tests and docs in the same patch whenever a public API changes.

## Docs

* **MUST** write docs in a tutorial voice for Quickstart/tutorial pages (your requirement).
* **MUST** prefer paragraphs over bullet points; bullet lists are allowed only when enumerating strict items (e.g., prerequisites, parameter lists).
* **MUST** follow the exact structure for each tutorial section:

  1. **What it is**
  2. **Why it exists / why designed this way**
  3. **How to use it** with concise example code
* **MUST** keep code examples runnable or explicitly labeled as pseudocode. If not runnable, **MUST** say what is omitted and why.
* **MUST** keep examples minimal: no large loops, no long configuration dumps, no unnecessary abstraction.
* **MUST** ensure docs match the current API: update doc snippets immediately when API changes (no stale examples).
* **MUST** include at least one “sanity check”/inspection step in examples when introducing new concepts (e.g., show how to print/inspect/validate objects).
* **MUST NOT** oversell: avoid visionary claims (“ecosystem”, “unprecedented”, etc.). Docs must state what the software does and how to use it.
* **MUST** keep terminology consistent: use the canonical names for objects and stages (e.g., “Atomistic”, “Topology”, “Frame”, “template”, “mapping”).

* **MUST** keep patches coherent: if you introduce a new concept, you must add the code + tests + docs entry in the same patch (no “TODO: docs later”).
* **MUST NOT** add broad refactors opportunistically while implementing a small feature unless explicitly requested.
