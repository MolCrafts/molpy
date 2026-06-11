<!-- mol-agent:bootstrap:managed begin -->
---
mol_project:
  name: molcrafts-molpy
  stage: experimental
  arch:
    style: layered
    rules_section: "Architecture Overview"
  docs:
    style: google
  science:
    required: true
  specs_path: .claude/specs
  notes_path: .claude/notes
  build:
    format: ruff format src tests
    check: ruff check src tests && ty check src/molpy/
    test: pytest tests/ -m "not external" -v
    test_single: pytest {} -v
---
<!-- mol-agent:bootstrap:managed end -->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Development Commands

```bash
# Setup
git clone https://github.com/MolCrafts/molpy.git
cd molpy
pip install -e ".[dev]"
pre-commit install

# Run tests
pytest tests/ -v -m "not external"                    # All local tests
pytest tests/test_core/ -v                             # Single module
pytest tests/test_core/test_atomistic.py::test_atom -v # Single test
pytest -k "pattern" -v                                 # Tests matching pattern
pytest --cov=src/molpy tests/ -v --cov-report=html    # With coverage

# Code quality
ruff format --check src tests     # Check formatting
ruff format src tests             # Auto-format
ruff check src tests              # Lint
ty check src/molpy/               # Type check
pre-commit run --all-files       # Run all pre-commit hooks

# Documentation
pip install -e ".[doc]"
mkdocs serve                      # Local preview at http://localhost:8000
mkdocs build                      # Build static site
```

## Architecture Overview

MolPy is a computational chemistry toolkit with explicit data flow and minimal magic. The core design philosophy:

- **Explicit data flow**: Few hidden side effects; the core data-model API mutates in place and returns `self` for chaining, with `.copy()` as the explicit opt-in for an independent object
- **Strong typing**: Public APIs use type hints; checked with `ty` (Astral)
- **Modular packages**: Each module has clear responsibility with minimal coupling

### Core Packages

| Package | Purpose |
|---------|---------|
| `core` | Data structures: `Entity`, `Link`, `Frame`, `Block`, `Atomistic`, `ForceField`. `Frame`/`Block` are backed by the molrs Rust column store (see below) |
| `io` | File I/O: readers/writers for PDB, GRO, LAMMPS DATA, XYZ, MOL2, AMBER (prmtop/inpcrd/prep/ac), GROMACS TOP, XSF, HDF5 formats |
| `parser` | Grammar-based parsing: SMILES, SMARTS, BigSMILES, GBigSMILES, CGSmiles |
| `builder` | System assembly: polymer builders, AmberTools integration, residue management |
| `typifier` | Atom typing: OPLS-AA, GAFF, custom SMARTS/SMIRKS-based typifiers |
| `compute` | Analysis: distance, angles, RDF, MSD, cross-correlation, and custom operators |
| `reacter` | Reaction framework: template-based reactions with leaving groups |
| `pack` | Packing workflows: Packmol integration, density targets |
| `engine` | MD abstractions: LAMMPS, CP2K, simulation management |
| `wrapper` | External tools: Antechamber, Prepgen, command-line wrappers |
| `adapter` | Format bridges: RDKit, OpenBabel, and other external libraries |

> **Hard runtime dependency**: `molcrafts-molrs` (Rust extension) is a required dependency declared in `pyproject.toml`. `Frame` and `Block` wrap `molrs.Frame` / `molrs.Block` — `core/frame.py` does `import molrs` at module load and all tabular data lives in the Rust Store. molpy does not run without it.

### Data Model Layer

The foundation is three class hierarchies:

1. **Entity** (dictionary-like): Base for all atoms, beads, and particles
   - Minimal, no ID management; identity-based hashing (`hash()` returns `id()`)
   - Subclasses: `Atom`, `Bead`

2. **Link** (connectivity): Holds ordered references to endpoints
   - Generic over endpoint type; endpoints are tuple of Entity
   - Subclasses: `Bond`, `Angle`, `Dihedral`, `Improper`, `CGBond`

3. **Struct** (topology container): Aggregates entities and links
   - Manages collections, provides selectors
   - Subclasses: `Atomistic`, `CoarseGrain`

**Block** and **Frame** are separate: tabular data (NumPy arrays) for fast computation.

4. **Frame** (numerical container): Holds named Blocks + metadata + box
   - `frame["atoms"]`, `frame["bonds"]` → Block objects
   - `frame.box` → `Box | None` (first-class attribute, **not** in metadata)
   - `frame.metadata` → arbitrary dict (timestep, format info, etc.)
   - `Block.rename(old, new)` for in-place column key rename

### Typical Workflow

```
1. Parse or build → Atomistic structure
2. Transform → reacter, builder, op modules
3. Typify → assign force-field types
4. Export or wrap → io, wrapper, engine modules
5. Analyze → compute operators
```

## Design Patterns & Extension Points

### Pattern: Adapter + Adapter Registry

For integrating external libraries (RDKit, LAMMPS, OpenBabel):
- Each integration is a separate adapter class under `adapter/` or `wrapper/`
- Adapters wrap the external tool with a consistent interface
- Optional import with fallback (don't force dependency unless needed)

**Example**: RDKit is optional; `RDKitAdapter` gracefully fails if not installed.

### Pattern: ForceField I/O

All force-field readers/writers inherit from base classes in `io.forcefield.base`:

```
ForceFieldReader (ABC)
  ├─ LAMMPSForceFieldReader
  ├─ XMLForceFieldReader
  └─ AmberPrmtopReader (reads AMBER prmtop/inpcrd)

ForceFieldWriter (ABC)
  ├─ LAMMPSForceFieldWriter
  ├─ XMLForceFieldWriter
  └─ ...
```

### Pattern: Formatter Hierarchy (`core.fields`)

Canonical field names and I/O boundary translation are defined in `core/fields.py`:

```
FieldSpec                              — canonical field definition (key, dtype, shape, doc)
    ↓
FieldFormatter                         — data field mapping: {format_key: FieldSpec}
    ↓                                     canonicalize() / localize() on Block
ForceFieldFormatter(FieldFormatter)    — inherits field mapping + param formatters: {StyleType: Callable}
```

**Per-format subclasses live in their own I/O module** (not centralized):

```python
# io/data/lammps.py
class LammpsFieldFormatter(FieldFormatter):
    _field_formatters = {"q": CHARGE, "mol": MOL_ID}

# io/forcefield/lammps.py
class LammpsForceFieldFormatter(LammpsFieldFormatter, ForceFieldFormatter):
    _param_formatters = {BondHarmonicStyle: _format_bond_harmonic, ...}
```

- Readers call `_formatter.canonicalize(block)` at exit (format → canonical)
- Writers call `_formatter.localize_frame(frame)` at entry (canonical → format, on a copy)
- `__init_subclass__` isolates registries per subclass
- `register_field()` / `register_param_formatter()` for runtime extension

**Canonical atom fields**: `charge` (not `q`), `mol_id` (not `mol`), `id`, `type`, `mass`, `x`/`y`/`z`, `element`, `symbol`, etc.

### Pattern: Mutation-Based Data Model + Explicit `.copy()`

**Critical**: The core data-model API mutates in place. `Atomistic`/`Struct` methods
`def_atom`, `def_bond`, `def_angle`, `def_dihedral`, `get_topo`, `move`, `rotate`,
`scale`, and `merge` all modify the structure in place and return `self` (or the
newly created entity) for method chaining. `.copy()` is the explicit opt-in for an
independent deep copy.

```python
# Building / transforming mutates in place and chains:
struct.def_atom(element="C", xyz=[0, 0, 0])   # adds atom, returns the Atom
struct.move([1, 0, 0], entity_type=Atom)      # mutates, returns self
struct.merge(other)                            # transfers other's entities into self

# When you need an independent object, copy explicitly:
work = struct.copy()        # deep copy; entities/links remapped
work.move([5, 0, 0], entity_type=Atom)   # struct is untouched
```

For *higher-level helper functions* (in `op`, `builder`, `reacter`, etc.), prefer
not mutating a caller-owned structure unexpectedly — `.copy()` first or build a new
structure and return it. This is a coding guideline for helpers, **not** how the
core `Atomistic`/`Struct`/`Frame` methods behave.

## Testing Guidelines

MolPy targets **80%+ code coverage**. All new code must have tests.

### Test Structure

Tests live in `tests/` mirroring `src/molpy/`:

```
tests/
├─ test_core/              # Data structures
├─ test_io/                # File I/O
├─ test_parser/            # Parsing
├─ test_builder/           # Builders
├─ test_typifier/          # Typifiers
├─ test_compute/           # Analysis
├─ test_reacter/           # Reactions
├─ test_wrapper/           # External tools
└─ test_engine/            # MD engines
```

### Marking External Tests

Tests requiring external executables (LAMMPS, Packmol, AmberTools) must be marked:

```python
import pytest

@pytest.mark.external
def test_lammps_integration():
    # requires LAMMPS executable
    pass
```

Run only local tests with: `pytest tests/ -m "not external"`

### Common Test Patterns

**In-place mutation + chaining checks** (core data-model API):
```python
def test_def_atom_mutates_and_returns_entity():
    struct = Atomistic()
    atom = struct.def_atom(element="C", xyz=[0, 0, 0])
    assert atom in struct.atoms          # added in place
    assert struct.move([1, 0, 0], entity_type=Atom) is struct  # returns self
```

**`.copy()` isolation checks** (when a helper must not mutate caller input):
```python
def test_helper_does_not_mutate_input():
    original = Atomistic()
    original.def_atom(element="C", xyz=[0, 0, 0])
    result = some_helper(original)        # helper does original.copy() internally
    assert result is not original         # independent object
    assert len(list(original.atoms)) == 1 # input untouched
```

**Adapter integration**:
```python
def test_adapter_fallback():
    # Test graceful failure if optional dep not installed
    adapter = RDKitAdapter(...)
    assert adapter is not None or rdkit_not_installed
```

## Key Module Notes

### `core.entity` and `core.atomistic`

- `Entity`: Base dict-like class for atoms/beads (no ID)
- `Link`: Generic container for bonds/angles/etc., holds `endpoints: tuple[Entity, ...]`
- `Atomistic`: Struct subclass managing atoms, bonds, angles, dihedrals
- All use identity-based hashing

### `core.frame` and `core.block`

- **Re-exported from molrs**: `frame.py` is a thin `from molrs import Block, Frame` re-export — `molpy.core.frame.Frame IS molrs.Frame` and `Block IS molrs.Block` (the rich Python layer in molrs over the Rust core). No molpy subclass, no `.to_molrs()` / `_inner` / `_source` bridge. `molcrafts-molrs` is a hard runtime dependency.
- `Block`: **numpy-only** typed columns (float / int / bool / str) in the Rust Store, exposed as zero-copy numpy views. There is **no Python-side object-column overflow** — a non-representable column (object / None / ragged) is rejected fail-fast at write (`molrs.BlockDtypeError` / `TypeError`).
- `Frame`: container of named Blocks + `metadata: dict` (Python-only annotations like timestep) + `box`. Built from a molrs world via the world's native `to_frame()` (e.g. `Atomistic.to_frame()` delegates to `molrs.Atomistic.to_frame()` and wraps the bare pyo3 frame with `Frame.from_dict` — zero Python-side densify/conversion).
- `Trajectory`: Sequence of Frames
- **Box is on `frame.box`**, never in `frame.metadata["box"]`

### `io.forcefield` hierarchy

- `LammpsForceFieldFormatter` inherits `LammpsFieldFormatter` + `ForceFieldFormatter`
- `_param_formatters` registry maps Style classes to serialization functions (subclass-isolated via `__init_subclass__`)
- Custom styles: subclass the formatter and register via `register_param_formatter()`
- Readers parse into `ForceField` objects; writers serialize via formatter dispatch

### `parser` module

- Grammar-based: uses Lark for SMILES/SMARTS/BigSMILES
- Grammar files in `parser/grammar/` and `parser/smiles/grammars/`
- Direct parser instantiation (not singleton)

### `builder` module

- Polymer builders: sequence generation, placement, crosslinking
- AmberTools integration: prepare molecules, run Antechamber, tleap
- All builders follow consistent factory/builder pattern

## Code Quality Standards

From `docs/developer/coding-style.md`:

- **Explicit over clever**: Clear behavior > shortcuts
- **Predictable mutation**: The core data model mutates in place; in higher-level helpers, don't mutate caller-owned structures unexpectedly — `.copy()` first or build and return a new structure
- **Small functions**: ~50 lines max; focused responsibility
- **Type hints**: Required on public APIs
- **Google-style docstrings**: For public functions/classes
- **Ruff formatting**: Non-negotiable (`ruff format`, enforced by pre-commit)

### Ready-to-commit checklist

- [ ] Code passes `ruff format --check` and `ruff check`
- [ ] Type check passes: `ty check src/molpy/`
- [ ] Tests cover changed behavior
- [ ] Public APIs have type hints and docstrings
- [ ] Helpers don't mutate caller-owned structures unexpectedly (`.copy()` when needed)
- [ ] No hardcoded values (use config or constants)
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`

## Common Gotchas

1. **Optional imports**: If adding a new external tool, follow the adapter pattern and test graceful fallback.
2. **Notebook output in git**: Pre-commit uses `nbstripout` to strip output; don't commit notebook output.
3. **External tools**: Mark tests with `@pytest.mark.external` if they need LAMMPS, Packmol, or AmberTools.
4. **Formatter registration**: Custom styles need `_param_formatters` registered on the format's `ForceFieldFormatter` subclass. Custom data fields need `_field_formatters` on the `FieldFormatter` subclass.
5. **Identity vs equality**: `Entity` and `Link` use identity-based hashing (`id(self)`), not value-based.
6. **Box is `frame.box`**: Never store box in `frame.metadata["box"]`. Use `frame.box` directly.
7. **Canonical field names**: Internal code uses `charge` (not `q`), `mol_id` (not `mol`). Readers/writers translate at the boundary via `FieldFormatter.canonicalize()`/`localize()`.

## Debugging Tips

- **Import errors in tests**: Reinstall with `pip install -e ".[dev]"` to ensure editable mode
- **Notebook doc build fails**: Run `pip install -e ".[doc]"` for doc deps
- **LAMMPS/Packmol tests fail**: Expected if executable not installed; use `-m "not external"`
- **Type checking**: Run locally with `ty check src/molpy/` (Astral's `ty`, also run in CI); config under `[tool.ty]` in `pyproject.toml`

---

## Skills & Agents

Workflow skills and review agents come from the **`mol` plugin** (molcrafts
claude-plugin marketplace). The former project-local `.claude/skills/` and
`.claude/agents/` were removed in favor of the plugin; project-specific knowledge
they carried now lives in `.claude/notes/` (architecture.md, performance.md,
testing.md, docs-style.md, ci.md).

### Skills (slash commands)

```bash
# Implementation workflow
/mol:spec "natural language need"   # NL → spec + binding acceptance contract (中文/English)
/mol:impl <spec>                    # TDD implementation from an approved spec (spec → tests → code → verify)
/mol:litrev "method or topic"       # Literature review before implementing physical models

# Documentation
/mol:docs [path]                    # Docstring audit/writing + narrative tutorials

# Validation
/mol:review [path]                  # Multi-axis code review (arch + perf + science + quality)
/mol:review --axis=arch             # Architecture layer dependency validation
/mol:review --axis=perf             # Performance / NumPy optimization review
/mol:test [path]                    # Test run, coverage analysis, scientific test audit

# CI & release
/mol:ci-sync                        # Audit/fix CI and pre-commit parity
/mol:ship merge                     # Pre-merge / release gate (PROCEED / BLOCK verdict)
```

### Agents

Review agents are provided by the mol plugin (`mol:architect`, `mol:scientist`,
`mol:tester`, `mol:documenter`, `mol:optimizer`, `mol:ci-guard`, …) and are
invoked through skills like `/mol:review`; release gating is handled by `/mol:ship`.

### Quick Start

```bash
# Implement a feature (full workflow)
/mol:spec "Add Morse bond potential"   # then: /mol:impl <spec>

# Start from vague requirements
/mol:spec "需要一个支持周期性边界条件的RDF计算器"

# Pre-PR validation
/mol:review
/mol:test
```
