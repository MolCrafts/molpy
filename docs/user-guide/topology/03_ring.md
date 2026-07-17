# Macrocycle

**Script:** [`examples/topology/03_ring.py`](../../../examples/topology/03_ring.py)

Ring digits in CGSmiles add one more residue edge. Bifunctional glycol is enough — the closing bond reuses free ends.

```python
ring = eo_builder().build_ring("EO", 6)
# → build("{[#EO]1[#EO][#EO][#EO][#EO][#EO]1}")
```

**Check:** for this condensation product, bond count equals atom count (one cycle).

```bash
cd examples && python topology/03_ring.py
```

## See also

- [Linear](01_linear.md) · [Section index](index.md)
