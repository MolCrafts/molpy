# Macrocycle

**Script:** [`examples/topology/03_ring.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/03_ring.py)

A ring bond closes the path: `{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}` bonds the last unit back to the first.

```python
import molpy as mp
from eo_kit import library

sites = mp.io.smiles.CGSmilesIR("{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}").to_coarsegrain()
ring = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

The growth placer lays the ring out as an open chain, so the closing bond starts at whatever distance its two ends grew apart; minimise before use.

**Check:** 6 units, 42 atoms, 42 bonds (one ring), no open port.

```bash
cd examples && python topology/03_ring.py
```

## See also

- [Star](04_star.md) · [Section index](index.md)
