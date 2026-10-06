# Star polymer

**Script:** [`examples/topology/04_star.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/04_star.py)

A star needs a **multifunctional core**: `X3` carries three `>` ports, one per arm. Arms are ordinary `EO` paths written as branches.

```python
import molpy as mp
from eo_kit import library

arm = "[#EO][#EO][#EO]"
sites = mp.io.CGSmilesIR(f"{{[#X3]({arm})({arm}){arm}}}").to_coarsegrain()
star = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

A core with fewer ports than arms is refused, naming the core site.

**Check:** 10 units, 77 atoms, 3 open ports at the arm ends.

```bash
cd examples && python topology/04_star.py
```

## See also

- [Comb](05_comb.md) · [Section index](index.md)
