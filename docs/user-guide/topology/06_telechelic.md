# Telechelic chain

**Script:** [`examples/topology/06_telechelic.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/06_telechelic.py)

End groups are one-port units at the ends of a path: `CAPA` (`C[>]`) joins the first `EO`'s `<`, and `CAPB` (`[<]OC`) joins the last `EO`'s `>`.

```python
import molpy as mp
from eo_kit import library

sites = mp.io.cgsmiles.CgSmilesIr("{[#CAPA][#EO]|6[#CAPB]}").to_coarsegrain()
tele = mp.builder.Assembler(library(), mp.builder.GrowthPlacer()).assemble(sites, mp.Atomistic)
```

**Check:** 8 units, 51 atoms, no open port.

```bash
cd examples && python topology/06_telechelic.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)
