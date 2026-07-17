# Comb polymer

**Script:** [`examples/topology/05_comb.py`](../../../examples/topology/05_comb.py)

Combs use multifunctional backbone units (`BR` in the kit) and **hand-written** CGSmiles through the sole entry `build` — irregular graphs that the `build_*` shortcuts do not cover.

```python
builder = eo_builder(extra={"BR": branch_unit()})
comb = builder.build("{[#EO][#BR]([#EO])[#EO][#BR]([#EO])[#EO]}")
# backbone EO–BR–EO–BR–EO with a one-unit graft on each BR
```

```bash
cd examples && python topology/05_comb.py
```

## See also

- [Star](04_star.md) · [Section index](index.md)
