# Comb polymer

**Script:** [`examples/topology/05_comb.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/05_comb.py)

Combs use multifunctional backbone units (`BR` in the kit) and **hand-written** CGSmiles through the sole entry `build` — irregular graphs that the `build_*` shortcuts do not cover.

```python
from eo_kit import branch_unit, eo_builder # examples/topology/
from molpy.builder.assembly import (
 CGSmilesBondIR,
 CGSmilesGraphIR,
 CGSmilesNodeIR,
)

# backbone EO–BR–EO–BR–EO with a one-unit graft on each BR
eo1, br1, g1 = (CGSmilesNodeIR(label=x) for x in ("EO", "BR", "EO"))
eo2, br2, g2 = (CGSmilesNodeIR(label=x) for x in ("EO", "BR", "EO"))
eo3 = CGSmilesNodeIR(label="EO")
topology = CGSmilesGraphIR(
 nodes=[eo1, br1, g1, eo2, br2, g2, eo3],
 bonds=[
 CGSmilesBondIR(node_i=eo1, node_j=br1),
 CGSmilesBondIR(node_i=br1, node_j=g1),
 CGSmilesBondIR(node_i=br1, node_j=eo2),
 CGSmilesBondIR(node_i=eo2, node_j=br2),
 CGSmilesBondIR(node_i=br2, node_j=g2),
 CGSmilesBondIR(node_i=br2, node_j=eo3),
 ],
)

builder = eo_builder(extra={"BR": branch_unit()})
comb = builder.build(topology)
```

```bash
cd examples && python topology/05_comb.py
```

## See also

- [Star](04_star.md) · [Section index](index.md)
