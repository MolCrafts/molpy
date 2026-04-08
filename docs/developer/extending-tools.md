# Adding a Tool or Compute Operation

This page shows how to add reusable recipes (`Tool`) and analysis operations (`Compute`) to MolPy.

## Which base class to use

| Need | Base class | Example |
|------|-----------|---------|
| Reusable multi-step recipe with configuration | `Tool` | `PrepareMonomer`, `BuildPolymer` |
| Reusable analysis on array data | `Compute` | `MSD`, `DisplacementCorrelation` |
| One-off calculation with no config | plain function | `compute_msd(positions)` |

Both `Tool` and `Compute` are frozen dataclasses. Configuration is set at construction and cannot change. Execution goes through `run()` (or `__call__`).

## Adding a Compute operation

Subclass `Compute`, declare configuration as frozen dataclass fields, implement `run()`.

```python
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from molpy.tool import Compute

@dataclass(frozen=True)
class RadiusOfGyration(Compute):
    """Compute radius of gyration for each frame."""

    use_masses: bool = True

    def run(self, positions: NDArray, masses: NDArray | None = None) -> NDArray:
        """Compute Rg for a set of positions.

        Args:
            positions: shape (n_atoms, 3)
            masses: shape (n_atoms,), optional

        Returns:
            Scalar Rg value.
        """
        if self.use_masses and masses is not None:
            w = masses / masses.sum()
        else:
            w = np.ones(len(positions)) / len(positions)

        com = (positions * w[:, None]).sum(axis=0)
        dr = positions - com
        rg2 = (w * (dr ** 2).sum(axis=1)).sum()
        return float(np.sqrt(rg2))
```

Usage:

```python
rg = RadiusOfGyration(use_masses=True)
value = rg(positions, masses)   # __call__ delegates to run()
```

## Adding a Tool recipe

Same pattern, but for `Tool` subclasses.

```python
from dataclasses import dataclass
from molpy.tool import Tool
from molpy.core.atomistic import Atomistic

@dataclass(frozen=True)
class ParameterizeMolecule(Tool):
    """Parse SMILES, generate 3D, typify, return typed structure."""

    force_field: str = "oplsaa"
    add_hydrogens: bool = True

    def run(self, smiles: str) -> Atomistic:
        import molpy as mp
        from molpy.typifier import OplsAtomisticTypifier

        mol = mp.parser.parse_molecule(smiles)
        mol = mp.tool.generate_3d(mol, add_hydrogens=self.add_hydrogens)
        mol.get_topo(gen_angle=True, gen_dihe=True)

        ff = mp.io.read_xml_forcefield(f"{self.force_field}.xml")
        typifier = OplsAtomisticTypifier(ff)
        return typifier.typify(mol)
```

Because the dataclass is frozen, the configuration cannot drift between calls. Two instances with different `force_field` values are two different protocols.

## Design rules

1. **Configuration goes in fields** — set once at init, frozen forever
2. **Runtime data goes through `run()`** — different inputs, same protocol
3. **No mutation** — `run()` returns new objects, never modifies inputs
4. **Keep `run()` focused** — one clear task, not a workflow engine
5. **Test in isolation** — each Tool/Compute should be testable with synthetic data

## Checklist

- [ ] Subclass `Tool` (recipe) or `Compute` (analysis)
- [ ] Add `@dataclass(frozen=True)` decorator
- [ ] Implement `run()` with type hints
- [ ] Write tests in `tests/test_tool/`
