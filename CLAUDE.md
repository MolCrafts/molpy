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
black --check src tests           # Check formatting
black src tests                   # Auto-format
pre-commit run --all-files       # Run all pre-commit hooks

# Documentation
pip install -e ".[doc]"
mkdocs serve                      # Local preview at http://localhost:8000
mkdocs build                      # Build static site
```

## Architecture Overview

MolPy is a computational chemistry toolkit with explicit data flow and minimal magic. The core design philosophy:

- **Explicit data flow**: No hidden side effects; transformations return new objects
- **Strong typing**: Public APIs use type hints; mypy-compatible
- **Modular packages**: Each module has clear responsibility with minimal coupling

### Core Packages

| Package | Purpose |
|---------|---------|
| `core` | Data structures: `Entity`, `Link`, `Frame`, `Block`, `Atomistic`, `ForceField` |
| `io` | File I/O: readers/writers for PDB, GRO, LAMMPS DATA, XYZ, JSON, HDF5 formats |
| `parser` | Grammar-based parsing: SMILES, SMARTS, BigSMILES, GBigSMILES, CGSmiles |
| `builder` | System assembly: polymer builders, AmberTools integration, residue management |
| `typifier` | Atom typing: OPLS-AA, GAFF, custom SMARTS/SMIRKS-based typifiers |
| `compute` | Analysis: distance, angles, RDF, MSD, cross-correlation, and custom operators |
| `reacter` | Reaction framework: template-based reactions with leaving groups |
| `pack` | Packing workflows: Packmol integration, density targets |
| `engine` | MD abstractions: LAMMPS, CP2K, simulation management |
| `wrapper` | External tools: Antechamber, Prepgen, command-line wrappers |
| `adapter` | Format bridges: RDKit, OpenBabel, and other external libraries |

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

### Pattern: Immutable Data Flow

**Critical**: Avoid mutation of input objects. Transformations return new objects:

```python
# WRONG
def add_hydrogens(mol):
    mol.add_atoms(...)  # mutates input

# CORRECT
def add_hydrogens(mol):
    new_mol = mol.copy()  # or rebuild from scratch
    # populate new_mol
    return new_mol
```

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

**Immutability checks**:
```python
def test_operation_does_not_mutate():
    original = Atomistic(...)
    result = some_operation(original)
    assert original is not result  # New object
    assert len(original.atoms) == original_count
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

- `Block`: Dict-like NumPy array container; auto-casts to ndarray; supports advanced indexing; `rename(old, new)` for column key rename
- `Frame`: Numerical container with named Blocks + `box: Box | None` (first-class attribute) + `metadata: dict`; `copy()` preserves box
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
- **Avoid mutation**: Return new objects, don't modify inputs
- **Small functions**: ~50 lines max; focused responsibility
- **Type hints**: Required on public APIs
- **Google-style docstrings**: For public functions/classes
- **Black formatting**: Non-negotiable (enforced by pre-commit)

### Ready-to-commit checklist

- [ ] Code passes `black --check`
- [ ] Tests cover changed behavior
- [ ] Public APIs have type hints and docstrings
- [ ] No mutation of input objects
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
- **Type checking**: No built-in mypy step yet; consider running locally: `mypy src/molpy`

---

## Skills & Agents

MolPy provides 9 skills (`.claude/skills/`) and 5 agents (`.claude/agents/`).

### Skills (slash commands)

```bash
# Implementation workflow
/molpy-impl "feature description"    # Full TDD workflow: litrev → spec → arch → test → code → review → docs
/molpy-spec "natural language need"  # NL → technical spec with literature grounding (中文/English)
/molpy-litrev "method or topic"      # Literature review before implementing physical models

# Documentation
/molpy-tutorial "concept or module"  # Write textbook-style User Guide page for human readers
/molpy-api-doc [path]                # Audit/write agent-friendly docstrings, type hints, unit annotations

# Validation
/molpy-arch [path]                   # Architecture layer dependency validation
/molpy-review [path]                 # Multi-dimensional code review (arch + perf + science + quality)
/molpy-test [path]                   # Test coverage analysis and scientific test audit
/molpy-perf [path]                   # Performance profiling and NumPy optimization review
```

### Agents (used by skills or directly)

| Agent | Domain | Tools |
|-------|--------|-------|
| `molpy-architect` | Layer dependencies, module design, pattern enforcement | Read, Grep, Glob, Bash |
| `molpy-scientist` | Equations, units, force field parameters, literature | Read, Grep, Glob, Bash, WebSearch, WebFetch |
| `molpy-tester` | TDD workflow, test design, scientific validation | Read, Grep, Glob, Bash, Write, Edit |
| `molpy-documenter` | Docstrings, unit annotations, scientific references | Read, Grep, Glob, Write, Edit |
| `molpy-optimizer` | NumPy vectorization, memory, algorithm complexity | Read, Grep, Glob, Bash |

### Quick Start

```bash
# Implement a feature (full workflow)
/molpy-impl "Add Morse bond potential"

# Start from vague requirements
/molpy-spec "需要一个支持周期性边界条件的RDF计算器"

# Pre-PR validation
/molpy-review --diff
/molpy-arch
/molpy-test
```
