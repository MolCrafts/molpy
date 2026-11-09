# Macrocycle

**Script:** [`molpy/examples/topology/03_ring.py`](https://github.com/MolCrafts/molpy/blob/dev/molpy/examples/topology/03_ring.py)

A ring bond closes the path: `{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}` bonds the last unit back to the first.

```python
import molpy as mp
from eo_kit import library

sites = mp.io.cgsmiles.read_cgsmiles("{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}").to_coarsegrain()
ring = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites)
```

The growth placer grows the ring from both ends, and the unit that closes it is placed by one rigid fit to both of its neighbours. Six rigid EO units cannot close a ring exactly: this fit misses its bond targets by about 0.66 Å root mean square, and a minimisation closes the rest. `GrowthPlacer(max_fit_rmsd=...)` turns a fit that misses by more into an error, for a structure whose geometry must close. Minimise before use.

**Check:** 6 units, 42 atoms, 42 bonds (one ring), no open port.

```bash
cd molpy/examples && python topology/03_ring.py
```

## See also

- [Star](04_star.md) · [Section index](index.md)
