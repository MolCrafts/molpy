# Comb polymer

**Script:** [`examples/topology/05_comb.py`](https://github.com/MolCrafts/molpy/blob/master/examples/topology/05_comb.py)

Combs use multifunctional backbone units (`BR` in the kit) and a **hand-written** residue topology through the sole entry `build` — irregular graphs that the `build_*` shortcuts do not cover.

```python
from eo_kit import branch_unit, eo_builder # examples/topology/
from molpy.builder.assembly import (
 ResidueBond,
 ResidueTopology,
 ResidueNode,
)

# backbone EO–BR–EO–BR–EO with a one-unit graft on each BR
eo1, br1, g1 = (ResidueNode(label=x) for x in ("EO", "BR", "EO"))
eo2, br2, g2 = (ResidueNode(label=x) for x in ("EO", "BR", "EO"))
eo3 = ResidueNode(label="EO")
topology = ResidueTopology(
 nodes=[eo1, br1, g1, eo2, br2, g2, eo3],
 bonds=[
 ResidueBond(node_i=eo1, node_j=br1),
 ResidueBond(node_i=br1, node_j=g1),
 ResidueBond(node_i=br1, node_j=eo2),
 ResidueBond(node_i=eo2, node_j=br2),
 ResidueBond(node_i=br2, node_j=g2),
 ResidueBond(node_i=br2, node_j=eo3),
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
