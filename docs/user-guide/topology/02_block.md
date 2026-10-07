# Block / sequence copolymer

**Script:** [`examples/topology/02_block.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/02_block.py)

A sequence is a path whose sites name different units. `{[#EO]|6[#PO]|4}` is six `EO` then four `PO`; any order can be written out site by site.

```python
import molpy as mp
from eo_kit import library

sites = mp.io.smiles.CGSmilesIR("{[#EO]|6[#PO]|4}").to_coarsegrain()
block = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

**Check:** 10 units, 84 atoms, 2 open ports.

```bash
cd examples && python topology/02_block.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)
