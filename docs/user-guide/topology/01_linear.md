# Linear homopolymer

**Script:** [`examples/topology/01_linear.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/01_linear.py)

A path of identical units is the simplest topology: `{[#EO]|10}` is ten `EO` sites, each bonded to the next.

```python
import molpy as mp
from eo_kit import library  # examples/topology/

sites = mp.io.cgsmiles.CgSmilesIr("{[#EO]|10}").to_coarsegrain()
chain = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

**Check:** 10 units (`frag_id` 0…9), 72 atoms (7 per unit plus the two end hydrogens), 71 bonds, 2 open ports at the ends.

```bash
cd examples && python topology/01_linear.py
```

## See also

- [Block / sequence](02_block.md) — different units on a path
- [Section index](index.md)
