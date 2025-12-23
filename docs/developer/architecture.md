# Architecture

This document describes MolPy's architecture, design principles, and package structure.

## Design Philosophy

MolPy is built on these core principles:

### 1. Composability

Small, focused components that work together:

```python
# Each component does one thing well
frame = read_pdb("protein.pdb")
topology = Topology(frame)
bonds = topology.bonds()
```

### 2. Explicit Over Implicit

Clear, predictable behavior:

```python
# Explicit: Clear what's happening
merged = frame1.merge(frame2)

# Not: Implicit modification
# frame1.merge(frame2)  # Does this modify frame1?
```

### 3. Type Safety

Full type hints throughout:

```python
def read_pdb(
    filepath: str | Path,
    model: int = 0
) -> Frame:
    """Read PDB file and return Frame."""
    ...
```

### 4. Immutability Where Possible

Core data structures behave like values:

```python
# Operations return new objects
frame2 = frame1.select("type == 'O'")
frame3 = frame2.translate([1, 0, 0])

# Original frame1 is unchanged
```

### 5. Separation of Concerns

- **Core** - Data structures (Frame, Block, Box)
- **IO** - Reading/writing files
- **Compute** - Calculations and analysis
- **Builder** - System construction
- **Engine** - External tool integration

## Package Structure

```
molpy/
├── core/           # Core data structures
├── io/             # File I/O
├── compute/        # Calculations and analysis
├── data/           # Force field data
├── engine/         # Simulation engines
├── optimize/       # Geometry optimization
├── wrapper/        # External tool wrappers
├── parser/         # String parsing (SMILES, SMARTS)
├── reacter/        # Chemical reactions
├── builder/        # System builders
├── pack/           # Molecular packing
├── potential/      # Potential functions
├── adapter/        # Toolkit adapters (RDKit)
├── typifier/       # Atom typing
└── op/             # Geometric operations
```

## Core Module

The `core` module provides fundamental data structures.

### Frame

The central data structure representing a molecular system:

```python
class Frame:
    """Container for molecular structure data.

    A Frame contains:
    - Named blocks of data (atoms, bonds, etc.)
    - Simulation box
    - Metadata
    """

    def __init__(self, box: Box | None = None):
        self._blocks: dict[str, Block] = {}
        self._box = box or Box()
        self._metadata: dict[str, Any] = {}
```

**Design decisions:**
- Blocks are accessed by name: `frame["atoms"]`
- Immutable operations return new Frame
- Box is always present (infinite if not specified)

### Block

Generic container for tabular data:

```python
class Block:
    """Container for columnar data.

    Like a lightweight DataFrame:
    - Named columns
    - Homogeneous length
    - NumPy-backed storage
    """

    def __init__(self, data: dict[str, ArrayLike]):
        self._data = {k: np.asarray(v) for k, v in data.items()}
```

**Design decisions:**
- Column-oriented storage
- NumPy arrays for efficiency
- No row-based indexing (use slicing)

### Box

Simulation box with periodic boundaries:

```python
class Box:
    """Simulation box for periodic boundaries.

    Supports:
    - Orthogonal boxes
    - Triclinic boxes
    - Infinite boxes
    """

    def __init__(
        self,
        lengths: Sequence[float] | None = None,
        angles: Sequence[float] | None = None
    ):
        ...
```

### Atomistic

Specialized structures for atoms, bonds, angles, dihedrals:

```python
class Atom:
    """Single atom representation."""
    id: int
    type: str
    position: np.ndarray
    mass: float

class Bond:
    """Bond between two atoms."""
    i: int  # Atom index
    j: int  # Atom index
    type: str | None
```

## IO Module

The `io` module handles file reading and writing.

### Design Pattern: Reader/Writer Classes

Each format has dedicated Reader and Writer classes:

```python
class PDBReader(DataReader):
    """Read PDB files."""

    def read(self, filepath: Path) -> Frame:
        ...

class PDBWriter(DataWriter):
    """Write PDB files."""

    def write(self, filepath: Path, frame: Frame) -> None:
        ...
```

### Factory Functions

Convenient functions wrap readers/writers:

```python
def read_pdb(filepath: str | Path, **kwargs) -> Frame:
    """Read PDB file."""
    reader = PDBReader(**kwargs)
    return reader.read(Path(filepath))
```

### Hierarchy

```
io/
├── data/           # Single-frame formats (PDB, XYZ, LAMMPS)
├── trajectory/     # Multi-frame formats
├── forcefield/     # Force field files
├── readers.py      # Factory functions for reading
└── writers.py      # Factory functions for writing
```

## Compute Module

The `compute` module provides analysis and calculations.

### Design: Functional Style

Computations are pure functions:

```python
def calculate_rdf(
    frame: Frame,
    r_max: float = 10.0,
    n_bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate radial distribution function.

    Returns:
        (r, g_r): Distance bins and RDF values
    """
    ...
```

### Result Objects

Complex results use dedicated classes:

```python
@dataclass
class RDFResult:
    """Result of RDF calculation."""
    r: np.ndarray
    g_r: np.ndarray
    n_bins: int
    r_max: float

    def plot(self) -> None:
        """Plot the RDF."""
        ...
```

## Wrapper vs Adapter Pattern

MolPy uses two distinct patterns for external integration:

### Wrapper Pattern

**Purpose:** Execute external command-line tools

**Example:** LAMMPS, Packmol, AmberTools

```python
class LammpsWrapper:
    """Wrapper for LAMMPS executable."""

    def run(
        self,
        script: Script,
        working_dir: Path
    ) -> subprocess.CompletedProcess:
        """Execute LAMMPS with given script."""
        cmd = [self.executable, "-in", script.path]
        return subprocess.run(cmd, cwd=working_dir)
```

**Characteristics:**
- Subprocess execution
- File-based communication
- Error handling for external failures

### Adapter Pattern

**Purpose:** Convert between MolPy and other Python libraries

**Example:** RDKit, MDAnalysis

```python
class RDKitAdapter:
    """Adapter between MolPy and RDKit."""

    @staticmethod
    def to_rdkit(frame: Frame) -> Chem.Mol:
        """Convert MolPy Frame to RDKit Mol."""
        ...

    @staticmethod
    def from_rdkit(mol: Chem.Mol) -> Frame:
        """Convert RDKit Mol to MolPy Frame."""
        ...
```

**Characteristics:**
- In-memory conversion
- Bidirectional (to/from)
- Type preservation

See [Wrapper & Adapter Patterns](wrapper_adapter_layers.ipynb) for detailed examples.

## Builder Module

The `builder` module constructs molecular systems.

### Design: Builder Pattern

Builders provide fluent interfaces:

```python
builder = PolymerBuilder()
polymer = (
    builder
    .add_monomer("CC(C)O", name="monomer1")
    .add_monomer("CCCO", name="monomer2")
    .set_chain_length(100)
    .set_sequence("AABB")
    .build()
)
```

### Hierarchy

```
builder/
├── crystal.py      # Crystal builders
└── polymer/        # Polymer builders
    ├── linear.py
    ├── branched.py
    └── crosslinked.py
```

## Reacter Module

The `reacter` module handles chemical reactions and topology modification.

### Design: Template-Based

Reactions are defined by templates:

```python
template = ReactionTemplate(
    reactants=["[C:1]=[C:2]", "[H:3][O:4]"],
    products=["[C:1][C:2]([O:4][H:3])"]
)

result = template.apply(frame)
```

### Components

- **Template** - Reaction definition
- **Selector** - Select reactive sites
- **Transformer** - Apply transformations
- **Connector** - Form new bonds

## Typifier Module

The `typifier` module assigns atom types.

### Design: Rule-Based Engine

Atom typing uses pattern matching:

```python
typifier = AtomTypifier(forcefield="oplsaa")
typed_frame = typifier.assign_types(frame)
```

### Layered Approach

1. **Graph matching** - Identify chemical environments
2. **Dependency analysis** - Resolve type dependencies
3. **Type assignment** - Assign final types

## Data Flow

Typical MolPy workflow:

```
Input File
    ↓
[IO Reader]
    ↓
Frame (core data structure)
    ↓
[Builder/Reacter] → Modified Frame
    ↓
[Typifier] → Typed Frame
    ↓
[Compute] → Analysis Results
    ↓
[IO Writer]
    ↓
Output File
```

## Extension Points

### Adding New File Format

1. Create reader class in `io/data/`:
```python
class MyFormatReader(DataReader):
    def read(self, filepath: Path) -> Frame:
        ...
```

2. Add factory function in `io/readers.py`:
```python
def read_myformat(filepath: str | Path) -> Frame:
    reader = MyFormatReader()
    return reader.read(Path(filepath))
```

3. Export in `io/__init__.py`

### Adding New Computation

1. Create function in `compute/`:
```python
def calculate_my_property(frame: Frame) -> np.ndarray:
    """Calculate my property."""
    ...
```

2. Export in `compute/__init__.py`

### Adding New Builder

1. Create builder class in `builder/`:
```python
class MySystemBuilder:
    def build(self) -> Frame:
        ...
```

2. Export in `builder/__init__.py`

## Performance Considerations

### NumPy Vectorization

Use NumPy operations instead of loops:

**Good:**
```python
distances = np.linalg.norm(positions[i] - positions[j], axis=1)
```

**Bad:**
```python
distances = [
    np.linalg.norm(positions[i] - positions[j])
    for i, j in pairs
]
```

### Lazy Evaluation

Defer expensive operations:

```python
class Topology:
    def __init__(self, frame: Frame):
        self._frame = frame
        self._bonds = None  # Compute on demand

    def bonds(self) -> list[Bond]:
        if self._bonds is None:
            self._bonds = self._detect_bonds()
        return self._bonds
```

### Memory Efficiency

Use views instead of copies when possible:

```python
# View (no copy)
subset = frame["atoms"]["x"][start:end]

# Copy (when needed)
subset = frame["atoms"]["x"][start:end].copy()
```

## Testing Architecture

### Unit Tests

Test individual components:
- `tests/test_core/` - Core data structures
- `tests/test_io/` - IO readers/writers
- `tests/test_compute/` - Computations

### Integration Tests

Test components working together:
- Read → Modify → Write roundtrips
- Builder → Typifier → Writer workflows

### Fixtures

Reusable test data:
```python
@pytest.fixture
def water_frame():
    """Provide water molecule frame."""
    return create_water_molecule()
```

## Further Reading

- [Wrapper & Adapter Patterns](wrapper_adapter_layers.ipynb)
- [Coding Style Guide](coding-style.md)
- [API Reference](../api/index.md)
