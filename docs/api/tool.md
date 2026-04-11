# Tool

High-level Tool workflows and analysis operations.

## Quick reference

### Tool workflows (polymer building)

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `PrepareMonomer` | BigSMILES → 3D Atomistic with ports and topology | Monomer template creation |
| `polymer(notation)` | Auto-detect notation → built chain | Quick single-chain building |
| `polymer_system(gbigsmiles)` | GBigSMILES → polydisperse chain list | Multi-chain system building |
| `generate_3d(mol)` | Add 3D coordinates via RDKit | Coordinate generation |

### Analysis (compute operations)

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `MSD` | Mean squared displacement | Diffusion analysis |
| `DisplacementCorrelation` | Cross-displacement correlation | Ion transport correlations |
| `compute_msd(positions)` | Convenience function for MSD | Quick MSD calculation |
| `compute_acf(series)` | Autocorrelation function | Time-series analysis |

## Canonical examples

```python
from molpy.tool import PrepareMonomer, polymer, generate_3d

# Prepare one monomer
prep = PrepareMonomer()
eo = prep.run("{[<]CCO[>]}")

# Build a chain (auto-detects notation)
chain = polymer("{[<]CCO[>]}|10|")

# Generate 3D coordinates
mol_3d = generate_3d(mol, add_hydrogens=True)
```

```python
from molpy.tool import MSD

msd = MSD(max_lag=3000)
msd_values = msd(unwrapped_positions)  # shape (max_lag,)
```

## Key behavior

- `Tool` subclasses are frozen dataclasses: configuration at init, execution via `.run()`
- `Compute` subclasses are callable: `compute(data)` or `compute.run(data)`
- `generate_3d` requires RDKit; raises `ImportError` if not installed

## Related

- [Concepts: Tool Layer](../tutorials/tools.md)
- [Guide: Polydisperse Systems](../user-guide/05_polydisperse_systems.ipynb)

---

## Full API

### Base

::: molpy.tool.base
    options:
      members:
        - Tool
        - Compute

### Polymer Tools

::: molpy.tool.polymer

### MSD

::: molpy.tool.msd

### Cross-Displacement Correlation

::: molpy.tool.cross_correlation

### Time Series

::: molpy.tool.time_series

### RDKit Tools

::: molpy.tool.rdkit
