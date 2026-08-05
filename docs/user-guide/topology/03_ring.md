# Macrocycle

**Script:** [`examples/topology/03_ring.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/03_ring.py)

Ring digits in CGSmiles add one more residue edge. Bifunctional glycol is enough — the closing bond reuses free ends.

```python
from eo_kit import eo_builder  # examples/topology/
from molpy.builder.assembly import ring_topology

ring = eo_builder().build_ring("EO", 6)
# → build(ring_topology(["EO"] * 6))
```

**Check:** for this condensation product, bond count equals atom count (one cycle).

```bash
cd examples && python topology/03_ring.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)
