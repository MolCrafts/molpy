# Star polymer

**Script:** [`examples/topology/04_star.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/04_star.py)

A star needs a **multifunctional core** (enough `fields.SITE` atoms for the arm count). Arms are ordinary bifunctional `EO`.

```python
from eo_kit import eo_builder, trifunctional_core

builder = eo_builder(extra={"X3": trifunctional_core()})
star = builder.build_star("X3", "EO", n_arms=3, arm_length=4)
# builds a star residue topology, then build(...)
```

Bifunctional monomers alone cannot branch: a bifunctional core has no free site for a third arm, so `build_star("EO", "EO", n_arms=3, ...)` raises `ValueError` naming the residue edge it could not form.

`placer=TracePlacer()` (set by `eo_builder`) grows each arm out of the core: every residue is moved rigidly next to the residue it hangs from, so all core–arm and arm–arm bonds start at bonding range.

```bash
cd examples && python topology/04_star.py
```

## See also

- [Comb](05_comb.md) · [Section index](index.md)
