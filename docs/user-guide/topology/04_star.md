# Star polymer

**Script:** [`examples/topology/04_star.py`](../../../examples/topology/04_star.py)

A star needs a **multifunctional core** (enough `fields.SITE` atoms for the arm count). Arms are ordinary bifunctional `EO`.

```python
from eo_kit import eo_builder, trifunctional_core

builder = eo_builder(extra={"X3": trifunctional_core()})
star = builder.build_star("X3", "EO", n_arms=3, arm_length=4)
# formats branched CGSmiles, then build(...)
```

Bifunctional monomers alone cannot branch: parentheses without extra sites collapse to a chain (by design).

```bash
cd examples && python topology/04_star.py
```

## See also

- [Comb](05_comb.md) · [Section index](index.md)
