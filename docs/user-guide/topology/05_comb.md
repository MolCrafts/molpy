# Comb polymer

**Script:** [`examples/topology/05_comb.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/05_comb.py)

A comb is a backbone whose branch points carry grafts. `BR` has the backbone ports `<` and `>` plus a graft port labelled `g`; the graft starts with `GR`, whose `<` carries the same label, so a graft can only join a graft port.

```python
import molpy as mp
from eo_kit import library

graft = "[#GR][#EO]"
sites = mp.io.smiles.CGSmilesIR(f"{{[#EO][#BR]({graft})[#EO][#BR]({graft})[#EO]}}").to_coarsegrain()
comb = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

**Check:** 9 units, 71 atoms, 4 open ports (two backbone ends, two graft ends).

```bash
cd examples && python topology/05_comb.py
```

## See also

- [Star](04_star.md) · [Section index](index.md)
