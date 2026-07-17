# Exhaustive crosslinked gel

**Script:** [`examples/topology/07_gel_exhaustive.py`](../../../examples/topology/07_gel_exhaustive.py)

Statistical network: grow strands, mark backbone carbons, replicate, then pair every allowed inter-chain site within a cutoff.

```text
build_linear → mark_backbone_crosslink_sites → Replicas.grid
  → GraphAssembler(XLINK).assemble(..., ExhaustiveSelector)
```

```python
from molpy.builder.assembly import ExhaustiveSelector, GraphAssembler, Replicas
from eo_kit import XLINK, eo_builder, mark_backbone_crosslink_sites
import molpy as mp

strand = eo_builder().build_linear("EO", 8)
mark_backbone_crosslink_sites(strand, step=2)
melt = Replicas(strand).grid(2, spacing=6.0, jitter=0.4, seed=3)

gel = GraphAssembler(mp.Reaction(XLINK)).assemble(
    melt,
    ExhaustiveSelector(cutoff=6.0, exclude_same_molecule=True),
)
```

`exclude_same_molecule=True` uses **bond connectivity** (not only `mol_id`) so intra-chain pairs are skipped while chains remain separate components.

```bash
cd examples && python topology/07_gel_exhaustive.py
```

## See also

- [Random gel](08_gel_random.md) · [Assembly](../02_assembly.md) · [Section index](index.md)
