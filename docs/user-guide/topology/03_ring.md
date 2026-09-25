# Macrocycle

**Script:** [`examples/topology/03_ring.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/03_ring.py)

A ring topology adds one more residue edge. Bifunctional glycol is enough — the closing bond reuses free ends.

```python
from eo_kit import eo_builder # examples/topology/
from molpy.builder.assembly import ring_topology

ring = eo_builder().build_ring("EO", 6)
# → build(ring_topology("EO", 6))
```

**Check:** for this condensation product, bond count equals atom count (one cycle).

**The closing bond is left long.** `TracePlacer` (set by `eo_builder`) walks the ring from residue 1 and moves each residue next to the residue it was reached from, so the five residue–residue bonds on that walk start at bonding range. The sixth bond, the one that closes the cycle between two residues the walk had already placed, is formed but not placed: it spans whatever distance the placement left between its two ends, and the script's `max length` line shows it. Shorten it with a geometry optimization before simulating (see [Geometry Optimization](../08_geometry_optimization.md)), or give the placer an explicit ring-shaped trace (`TracePlacer().with_trace(Trace(points))`, see [Assembly](../02_assembly.md#giving-the-chain-a-shape)).

```bash
cd examples && python topology/03_ring.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)
