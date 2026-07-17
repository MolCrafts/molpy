# Random crosslinked gel

**Script:** [`examples/topology/08_gel_random.py`](../../../examples/topology/08_gel_random.py)

Same melt as the exhaustive gel; pairing is `RandomSelector` to a target **conversion** (fraction of the limiting reactant’s sites) with a **seed** for reproducibility.

```python
from molpy.builder.assembly import GraphAssembler, RandomSelector, Replicas
import molpy as mp
from eo_kit import XLINK, eo_builder, mark_backbone_crosslink_sites

strand = eo_builder().build_linear("EO", 8)
mark_backbone_crosslink_sites(strand, step=2)
melt = Replicas(strand).grid(2, spacing=6.0, jitter=0.4, seed=3)

gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
    melt,
    RandomSelector(
        conversion=0.5,
        seed=7,
        cutoff=6.0,
        exclude_same_molecule=True,
    ),
)
# same seed → same atom count
```

```bash
cd examples && python topology/08_gel_random.py
```

## See also

- [Exhaustive gel](07_gel_exhaustive.md) · [Section index](index.md)
